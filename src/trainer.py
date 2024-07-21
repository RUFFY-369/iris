# from collections import defaultdict
# from functools import partial
# from pathlib import Path
# import shutil
# import sys
# import time
# from typing import Any, Dict, Optional, Tuple, List

# from einops import rearrange
# from envs.world_model_env import WorldModelEnv


# import random
# import hydra
# from hydra.utils import instantiate
# from omegaconf import DictConfig, OmegaConf
# import torch
# import torch.nn as nn
# from tqdm import tqdm
# import wandb

# from agent import Agent
# from collector import Collector
# from envs import SingleProcessEnv, MultiProcessEnv
# from episode import Episode
# from make_reconstructions import make_reconstructions_from_batch
# from models.actor_critic import ActorCritic
# from models.world_model import WorldModel
# from utils import configure_optimizer, EpisodeDirManager, set_seed
# import transformers as t

# global_rng = random.Random()


# def ids_tensor(shape, vocab_size, rng=None, name=None):
#     #  Creates a random int32 tensor of the shape within the vocab size
#     if rng is None:
#         rng = global_rng

#     total_dims = 1
#     for dim in shape:
#         total_dims *= dim

#     values = []
#     for _ in range(total_dims):
#         values.append(rng.randint(0, vocab_size - 1))

#     return torch.tensor(data=values, dtype=torch.long, device="cpu").view(shape).contiguous()


# def random_attention_mask(shape, rng=None, name=None):
#     attn_mask = ids_tensor(shape, vocab_size=2, rng=None, name=None)
#     # make sure that at least one token is attended to for each batch
#     # we choose the 1st token so this property of `at least one being non-zero` still holds after applying causal mask
#     attn_mask[:, 0] = 1
#     return attn_mask


# def floats_tensor(shape, scale=1.0, rng=None, name=None):
#     """Creates a random float32 tensor"""
#     if rng is None:
#         rng = global_rng

#     total_dims = 1
#     for dim in shape:
#         total_dims *= dim

#     values = []
#     for _ in range(total_dims):
#         values.append(rng.random() * scale)

#     return torch.tensor(data=values, dtype=torch.float, device="cpu").view(shape).contiguous()

# class Trainer:
#     def __init__(self, cfg: DictConfig) -> None:
#         wandb.init(
#             config=OmegaConf.to_container(cfg, resolve=True),
#             reinit=True,
#             resume=True,
#             **cfg.wandb
#         )

#         if cfg.common.seed is not None:
#             set_seed(cfg.common.seed)

#         self.cfg = cfg
#         self.start_epoch = 1
#         self.device = torch.device(cfg.common.device)

#         self.ckpt_dir = Path('checkpoints')
#         self.media_dir = Path('media')
#         self.episode_dir = self.media_dir / 'episodes'
#         self.reconstructions_dir = self.media_dir / 'reconstructions'

#         if not cfg.common.resume:
#             config_dir = Path('config')
#             config_path = config_dir / 'trainer.yaml'
#             config_dir.mkdir(exist_ok=False, parents=False)
#             shutil.copy('.hydra/config.yaml', config_path)
#             wandb.save(str(config_path))
#             shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "src"), dst="./src")
#             shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "scripts"), dst="./scripts")
#             self.ckpt_dir.mkdir(exist_ok=False, parents=False)
#             self.media_dir.mkdir(exist_ok=False, parents=False)
#             self.episode_dir.mkdir(exist_ok=False, parents=False)
#             self.reconstructions_dir.mkdir(exist_ok=False, parents=False)

#         episode_manager_train = EpisodeDirManager(self.episode_dir / 'train', max_num_episodes=cfg.collection.train.num_episodes_to_save)
#         episode_manager_test = EpisodeDirManager(self.episode_dir / 'test', max_num_episodes=cfg.collection.test.num_episodes_to_save)
#         self.episode_manager_imagination = EpisodeDirManager(self.episode_dir / 'imagination', max_num_episodes=cfg.evaluation.actor_critic.num_episodes_to_save)

#         def create_env(cfg_env, num_envs):
#             env_fn = partial(instantiate, config=cfg_env)
#             return MultiProcessEnv(env_fn, num_envs, should_wait_num_envs_ratio=1.0) if num_envs > 1 else SingleProcessEnv(env_fn)

#         if self.cfg.training.should:
#             train_env = create_env(cfg.env.train, cfg.collection.train.num_envs)
#             self.train_dataset = instantiate(cfg.datasets.train)
#             self.train_collector = Collector(train_env, self.train_dataset, episode_manager_train)

#         if self.cfg.evaluation.should:
#             test_env = create_env(cfg.env.test, cfg.collection.test.num_envs)
#             self.test_dataset = instantiate(cfg.datasets.test)
#             self.test_collector = Collector(test_env, self.test_dataset, episode_manager_test)

#         assert self.cfg.training.should or self.cfg.evaluation.should
#         env = train_env if self.cfg.training.should else test_env

#         tokenizer = instantiate(cfg.tokenizer)
#         # print("actions",env.num_actions)
#         world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=env.num_actions, config=instantiate(cfg.world_model))
#         actor_critic = ActorCritic(**cfg.actor_critic, act_vocab_size=env.num_actions)
#         self.agent = Agent(tokenizer, world_model, actor_critic).to(self.device)
#         # for name, param in self.agent.named_parameters():
#         #     print("AGENTTTT",name, param.shape)
#         # for name, param in world_model.named_parameters():
#         #     print("WORLD MODEL",name, param.shape)
#         # for name, param in actor_critic.named_parameters():
#         #     print("ACTOR CRITIC",name, param.shape)
#         # for name, param in tokenizer.named_parameters():
#         #     print("TOKENIZER",name, param.shape)
#         print(f'{sum(p.numel() for p in self.agent.tokenizer.parameters())} parameters in agent.tokenizer')
#         print(f'{sum(p.numel() for p in self.agent.world_model.parameters())} parameters in agent.world_model')
#         print(f'{sum(p.numel() for p in self.agent.actor_critic.parameters())} parameters in agent.actor_critic')

#         self.optimizer_tokenizer = torch.optim.Adam(self.agent.tokenizer.parameters(), lr=cfg.training.learning_rate)
#         self.optimizer_world_model = configure_optimizer(self.agent.world_model, cfg.training.learning_rate, cfg.training.world_model.weight_decay)
#         self.optimizer_actor_critic = torch.optim.Adam(self.agent.actor_critic.parameters(), lr=cfg.training.learning_rate)

#         if cfg.initialization.path_to_checkpoint is not None:
#             self.agent.load(**cfg.initialization, device=self.device)

#         if cfg.common.resume:
#             self.load_checkpoint()

#     def run(self) -> None:

#         for epoch in range(self.start_epoch, 1 + self.cfg.common.epochs):

#             print(f"\nEpoch {epoch} / {self.cfg.common.epochs}\n")
#             start_time = time.time()
#             to_log = []

#             if self.cfg.training.should:
#                 if epoch <= self.cfg.collection.train.stop_after_epochs:
#                     to_log += self.train_collector.collect(self.agent, epoch, **self.cfg.collection.train.config)
#                 to_log += self.train_agent(epoch)

#             # if self.cfg.evaluation.should and (epoch % self.cfg.evaluation.every == 0):
#             #     self.test_dataset.clear()
#             #     to_log += self.test_collector.collect(self.agent, epoch, **self.cfg.collection.test.config)
#             #     to_log += self.eval_agent(epoch)

#             if self.cfg.training.should:
#                 self.save_checkpoint(epoch, save_agent_only=not self.cfg.common.do_checkpoint)

#             to_log.append({'duration': (time.time() - start_time) / 3600})
#             for metrics in to_log:
#                 wandb.log({'epoch': epoch, **metrics})

#         self.finish()

#     def train_agent(self, epoch: int) -> None:
#         self.agent.eval() #SETS THE MODEL'S MODE TO TRAIN BUT DOESNT REALLY TRAIN THE MODEL(personal comment to remember)
#         self.agent.zero_grad()

#         metrics_tokenizer, metrics_world_model, metrics_actor_critic = {}, {}, {}

#         self.cfg_tokenizer = self.cfg.training.tokenizer
#         self.cfg_world_model = self.cfg.training.world_model
#         self.cfg_actor_critic = self.cfg.training.actor_critic
#         # print("cfg,toke",type(cfg_tokenizer) )

#         if epoch > self.cfg_tokenizer.start_after_epochs:
#             print("in tokkk")
#             metrics_tokenizer = self.train_component(self.agent.tokenizer, self.optimizer_tokenizer, self.cfg_tokenizer,self.cfg_world_model,self.cfg_actor_critic)
#         self.agent.tokenizer.eval()

#         if epoch > self.cfg_world_model.start_after_epochs:
#             metrics_world_model = self.train_component(self.agent.world_model, self.optimizer_world_model, sequence_length=self.cfg.common.sequence_length, sample_from_start=True, tokenizer=self.agent.tokenizer, **cfg_world_model)
#         self.agent.world_model.eval()

#         if epoch > self.cfg_actor_critic.start_after_epochs:
#             metrics_actor_critic = self.train_component(self.agent.actor_critic, self.optimizer_actor_critic, sequence_length=1 + self.cfg.training.actor_critic.burn_in, sample_from_start=False, tokenizer=self.agent.tokenizer, world_model=self.agent.world_model, **cfg_actor_critic)
#         self.agent.actor_critic.eval()

#         return [{'epoch': epoch, **metrics_tokenizer, **metrics_world_model, **metrics_actor_critic}]

#     def train_component(self, component: nn.Module, optimizer: torch.optim.Optimizer, tok,wm,ac) -> Dict[str, float]:
#         loss_total_epoch = 0.0
#         intermediate_losses = defaultdict(float)

#         for _ in tqdm(range(200), desc=f"Training {str(component)}", file=sys.stdout):
#             optimizer.zero_grad()
#             for _ in range(1):
#                 # batch_t = self.train_dataset.sample_batch(16, 1, True)
#                 # batch_wm = self.train_dataset.sample_batch(4, 20, True)
#                 # batch_ac = self.train_dataset.sample_batch(4, 21, False)
#                 # batch_t = self._to_device(batch_t)
#                 # batch_wm = self._to_device(batch_wm)
#                 # batch_ac = self._to_device(batch_ac)
#                 torch.manual_seed(0)
#                 config = t.IrisConfig()
#                 # with torch.no_grad():

#                 observations_tokenizer = floats_tensor((3,1,config.in_channels,config.resolution,config.resolution))
#                 actions_tokenizer = ids_tensor((3,1),vocab_size =4).long()
#                 rewards_tokenizer = ids_tensor((3,1),vocab_size =8)
#                 zeros = torch.zeros_like(rewards_tokenizer)
#                 # Rewards are given depending on which color brick is broken in 'Breakout' Atari env
#                 zeros[((rewards_tokenizer==7)|( rewards_tokenizer==4)|(rewards_tokenizer==1))]=1
#                 rewards_tokenizer = torch.mul(zeros,rewards_tokenizer).float()
#                 ends_tokenizer = torch.zeros(3,1).long()
#                 for i in range(3):
#                     ends_tokenizer[i,ids_tensor((1,),vocab_size=1).item()]=1 if floats_tensor((1,)).item()<0.5 else 0
#                 mask_padding_tokenizer = torch.ones(3,1).bool()

#                 observations_world_model = floats_tensor((3,20,config.in_channels,config.resolution,config.resolution))
#                 actions_world_model = ids_tensor((3,20),vocab_size =4).long()
#                 rewards_world_model = ids_tensor((3,20),vocab_size =8)
#                 zeros = torch.zeros_like(rewards_world_model)
#                 # Rewards are given depending on which color brick is broken in 'Breakout' Atari env
#                 zeros[((rewards_world_model==7)|( rewards_world_model==4)|(rewards_world_model==1))]=1
#                 rewards_world_model = torch.mul(zeros,rewards_world_model).float()
#                 ends_world_model = torch.zeros(3,20).long()
#                 for i in range(3):
#                     ends_world_model[i,ids_tensor((1,),vocab_size=20).item()]=1 if floats_tensor((1,)).item()<0.5 else 0
#                 mask_padding_world_model = torch.ones(3,20).bool()

#                 observations_actor_critic = floats_tensor((3,21,config.in_channels,config.resolution,config.resolution))
#                 actions_actor_critic = ids_tensor((3,21),vocab_size =4).long()
#                 rewards_actor_critic = ids_tensor((3,21),vocab_size =8)
#                 zeros = torch.zeros_like(rewards_actor_critic)
#                 # Rewards are given depending on which color brick is broken in 'Breakout' Atari env
#                 zeros[((rewards_actor_critic==7)|( rewards_actor_critic==4)|(rewards_actor_critic==1))]=1
#                 rewards_actor_critic = torch.mul(zeros,rewards_actor_critic).float()
#                 ends_actor_critic = torch.zeros(3,21).long()
#                 for i in range(3):
#                     ends_actor_critic[i,ids_tensor((1,),vocab_size=21).item()]=1 if floats_tensor((1,)).item()<0.5 else 0
#                 mask_padding_actor_critic = torch.ones(3,21).bool()

#                 # print("MASKKKKKKKKKKKKKKK",mask_padding_actor_critic[:, -1].all(),mask_padding_actor_critic[:, -1])
                
#                 observations = [observations_tokenizer,observations_world_model,observations_actor_critic]
#                 actions = [actions_tokenizer,actions_world_model,actions_actor_critic]
#                 rewards = [rewards_tokenizer,rewards_world_model,rewards_actor_critic]
#                 ends = [ends_tokenizer,ends_world_model,ends_actor_critic]
#                 mask_padding = [mask_padding_tokenizer,mask_padding_world_model,mask_padding_actor_critic]


#                 batch_t = dict(observations = observations[0], actions = actions[0], rewards = rewards[0], ends = ends[0], mask_padding = mask_padding[0])
#                 batch_wm = dict(observations = observations[1], actions = actions[1], rewards = rewards[1], ends = ends[1], mask_padding = mask_padding[1])
#                 batch_ac = dict(observations = observations[2], actions = actions[2], rewards = rewards[2], ends = ends[2], mask_padding = mask_padding[2])
                
#                 batch_t = self._to_device(batch_t)
#                 batch_wm = self._to_device(batch_wm)
#                 batch_ac = self._to_device(batch_ac)
#                 print("SHAPEEEEEEEEE", batch_t["observations"].shape,batch_t["actions"].shape,batch_t["rewards"].shape,batch_t["ends"].shape,
#                         batch_t["mask_padding"].shape)
#                 print("SHAPEEEEEEEEE2", batch_wm["observations"].shape,batch_wm["actions"].shape,batch_wm["rewards"].shape,batch_wm["ends"].shape,
#                         batch_wm["mask_padding"].shape)
#                 print("SHAPEEEEEEEEE3", batch_ac["observations"].shape,batch_ac["actions"].shape,batch_ac["rewards"].shape,batch_ac["ends"].shape,
#                         batch_ac["mask_padding"].shape)
#                 # SHAPEEEEEEEEE torch.Size([32, 1, 3, 64, 64]) torch.Size([32, 1]) torch.Size([32, 1]) torch.Size([32, 1]) torch.Size([32, 1])
#                 # SHAPEEEEEEEEE2 torch.Size([8, 20, 3, 64, 64]) torch.Size([8, 20]) torch.Size([8, 20]) torch.Size([8, 20]) torch.Size([8, 20])
#                 # SHAPEEEEEEEEE3 torch.Size([8, 21, 3, 64, 64]) torch.Size([8, 21]) torch.Size([8, 21]) torch.Size([8, 21]) torch.Size([8, 21])
#                 print("obsssssss",batch_wm["ends"].bool(),batch_wm["ends"],torch.all(batch_wm["ends"].sum(dim=1) <= 1),batch_wm["ends"].sum(dim=1))
                
#                 model = t.IrisModel.from_pretrained("ruffy369/iris-breakout")
#                 model.eval()
                
#                 output_hf_dict = model(observations = [batch_t['observations'],batch_wm['observations'],batch_ac['observations']],
#                                 actions = [batch_t['actions'],batch_wm['actions'],batch_ac['actions']],
#                                 rewards = [batch_t['rewards'],batch_wm['rewards'],batch_ac['rewards']],
#                                 ends = [batch_t['ends'],batch_wm['ends'],batch_ac['ends']],
#                                 mask_padding = [batch_t['mask_padding'],batch_wm['mask_padding'],batch_ac['mask_padding']],
#                                 should_preprocess = False,
#                                 should_postprocess = False,
#                                 output_hidden_states = True,
#                                 output_attentions = True,
                                
#                                 )

                
                        
                
#                 # hs  = output_hf_dict.attentions[0]
#                 # hs.retain_grad()
#                 # output_hf_dict[0].flatten()[0].backward(retain_graph=True)
#                 # print("GRADDDDDDDDDDDDDDd", hs.grad)
     
        
#                 print("losssssssss",output_hf_dict.obs_preds.shape)

#                 # time.sleep(6)
#                 # print("appppppppppppppp", output_hf_dict.action_preds)
#                 # time.sleep(6)
#                 # print("rppppppppppppppp", output_hf_dict.reward_preds)
#                 # time.sleep(6)
#                 # print("eeeeeeeeeeeeeeeeee", output_hf_dict.epsiode_end)
#                 # time.sleep(6)
#                 # print("obsssssssssss", output_hf_dict.obs_preds)
                
#                     # print("OUTPUTTTTTTTTT:", output_hf.hidden_states[0][0].shape,output_hf.hidden_states[0][1].shape,output_hf.hidden_states[1][0].shape,
#                     #       output_hf.hidden_states[1][1].shape,output_hf.hidden_states[1][2].shape,output_hf.hidden_states[1][3].shape,
#                     #       output_hf.hidden_states[2].shape)
#                     # #torch.Size([32, 1, 3, 64, 64])  torch.Size([8, 1, 4]) torch.Size([8, 20, 3]) torch.Size([8, 20, 2]) torch.Size([8, 320, 512])
#                     # a = output_hf.obs_preds
                
#                     # time.sleep(3)
#                     # batch = self.train_dataset.sample_batch(batch_num_samples, sequence_length, sample_from_start)
#                     # batch = self._to_device(batch)
#                     # output_hf = model(observations = [batch_t['observations'],batch_wm['observations'],batch_ac['observations']],
#                     #                 actions = [batch_t['actions'],batch_wm['actions'],batch_ac['actions']],
#                     #                 rewards = [batch_t['rewards'],batch_wm['rewards'],batch_ac['rewards']],
#                     #                 ends = [batch_t['ends'],batch_wm['ends'],batch_ac['ends']],
#                     #                 mask_padding = [batch_t['mask_padding'],batch_wm['mask_padding'],batch_ac['mask_padding']],
#                     #                 should_preprocess = False,
#                     #                 should_postprocess = False,
#                     #                 output_hidden_states = True,
#                     #                 output_attentions = False,
#                     #                 return_dict = False)
                
#                 self.agent.tokenizer.eval()
#                 self.agent.world_model.eval()
#                 self.agent.actor_critic.eval()
#                 # for i in range(10):
#                 # print("BOTHHH OUTPRRRRRRRR", output_hf
#                 #       )
#                 # time.sleep(30)
#                 # output_hf_dict = output_hf_dict.to_tuple()
#                 # output_hf_dict[output_hf_dict != output_hf_dict] = 0
#                 # output_hf[output_hf != output_hf] = 0
                
#                 # print("nowwwwwwwwww", torch.allclose(output_hf, output_hf_dict))
                    
                    
#                 print("NOW REAL WILL USE") 

#                 # CHECK LOSS
#                 # losses_tokenizer = self.agent.tokenizer.compute_loss(batch_t, **self.cfg_tokenizer)
#                 # losses_tokenizer = losses_tokenizer.loss_total
#                 # losses_world_model = self.agent.world_model.compute_loss(batch_wm, self.agent.tokenizer,**self.cfg_world_model)
#                 # losses_world_model = losses_world_model.loss_total
#                 losses_actor_critic = self.agent.actor_critic.compute_loss(batch_ac, self.agent.tokenizer,self.agent.world_model,**self.cfg_actor_critic)
#                 losses_actor_critic = losses_actor_critic.loss_total
#                 print("losssssssss",losses_actor_critic)
                
#                 #CHECK OUTPUT PRECISION
#                 # out_tok = self.agent.tokenizer.forward(batch_t['observations'], True, True)  
#                 #     # losses = self.agent.tokenizer.compute_loss(batch_t, **tok) 
#                 #     # print("out tok", losses[0],losses[1],losses[2])
#                 # with torch.no_grad():
#                 #     obs_tokens = self.agent.tokenizer.encode(batch_wm['observations'], should_preprocess=True).tokens
#                 #     # print("TOKENSSSS ZISEEE", obs_tokens.shape)
#                 # act_tokens = rearrange(batch_wm['actions'], 'b l -> b l 1')
#                 # tokens = rearrange(torch.cat((obs_tokens, act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L(K+1)) 
#                 # # print("TOKENSSS REAL", batch_wm['observations'])
#                 # out_wm = self.agent.world_model.forward(tokens) 
#                 # # print("out wm", losses.logits_observations,losses.logits_observations.shape)

#                 # wm_env = WorldModelEnv(self.agent.tokenizer, self.agent.world_model, batch_ac['observations'].device)
#                 # obs = wm_env.reset_from_initial_observations(batch_ac['observations'][:, -1])
#                 # self.agent.actor_critic.reset(n=batch_ac['observations'].size(0))
#                 # out_ac = self.agent.actor_critic.forward(obs) 

#                 # print("distanceeeeeeeeee",torch.dist(output_hf.action_preds,out_ac.logits_actions))
#                 # print("ACTIONSSSSSSSSSSS",output_hf[0],output_hf.losses,output_hf[1],output_hf["reconstructed_img"])
#                 # time.sleep(2)
#                     # print(self.agent.state_dict())
#                     # print(model.state_dict())
#                     # for key in self.agent.state_dict().keys():
#                 #     print(torch.allclose(model.state_dict()[key],self.agent.state_dict()[key]))
#                 # loss_total_step.backward()
#                 # loss_total_epoch += loss_total_step.item() / steps_per_epoch

#                 for loss_name, loss_value in losses.intermediate_losses.items():
#                     intermediate_losses[f"{str(component)}/train/{loss_name}"] += loss_value / 200

#             # if 10.0 is not None:
#             torch.nn.utils.clip_grad_norm_(component.parameters(), 10.0)

#             optimizer.step()

#         metrics = {f'{str(component)}/train/total_loss': loss_total_epoch, **intermediate_losses}
#         return metrics

#     @torch.no_grad()
#     def eval_agent(self, epoch: int) -> None:
#         self.agent.eval()

#         metrics_tokenizer, metrics_world_model = {}, {}

#         cfg_tokenizer = self.cfg.evaluation.tokenizer
#         cfg_world_model = self.cfg.evaluation.world_model
#         cfg_actor_critic = self.cfg.evaluation.actor_critic

#         if epoch > cfg_tokenizer.start_after_epochs:
#             metrics_tokenizer = self.eval_component(self.agent.tokenizer, cfg_tokenizer.batch_num_samples, sequence_length=1)

#         if epoch > cfg_world_model.start_after_epochs:
#             metrics_world_model = self.eval_component(self.agent.world_model, cfg_world_model.batch_num_samples, sequence_length=self.cfg.common.sequence_length, tokenizer=self.agent.tokenizer)

#         if epoch > cfg_actor_critic.start_after_epochs:
#             self.inspect_imagination(epoch)

#         if cfg_tokenizer.save_reconstructions:
#             batch = self._to_device(self.test_dataset.sample_batch(batch_num_samples=3, sequence_length=self.cfg.common.sequence_length))
#             make_reconstructions_from_batch(batch, save_dir=self.reconstructions_dir, epoch=epoch, tokenizer=self.agent.tokenizer)

#         return [metrics_tokenizer, metrics_world_model]

#     @torch.no_grad()
#     def eval_component(self, component: nn.Module, batch_num_samples: int, sequence_length: int, **kwargs_loss: Any) -> Dict[str, float]:
#         loss_total_epoch = 0.0
#         intermediate_losses = defaultdict(float)

#         steps = 0
#         pbar = tqdm(desc=f"Evaluating {str(component)}", file=sys.stdout)
#         for batch in self.test_dataset.traverse(batch_num_samples, sequence_length):
#             batch = self._to_device(batch)

#             losses = component.compute_loss(batch, **kwargs_loss)
#             loss_total_epoch += losses.loss_total.item()

#             for loss_name, loss_value in losses.intermediate_losses.items():
#                 intermediate_losses[f"{str(component)}/eval/{loss_name}"] += loss_value

#             steps += 1
#             pbar.update(1)

#         intermediate_losses = {k: v / steps for k, v in intermediate_losses.items()}
#         metrics = {f'{str(component)}/eval/total_loss': loss_total_epoch / steps, **intermediate_losses}
#         return metrics

#     @torch.no_grad()
#     def inspect_imagination(self, epoch: int) -> None:
#         mode_str = 'imagination'
#         batch = self.test_dataset.sample_batch(batch_num_samples=self.episode_manager_imagination.max_num_episodes, sequence_length=1 + self.cfg.training.actor_critic.burn_in, sample_from_start=False)
#         outputs = self.agent.actor_critic.imagine(self._to_device(batch), self.agent.tokenizer, self.agent.world_model, horizon=self.cfg.evaluation.actor_critic.horizon, show_pbar=True)

#         to_log = []
#         for i, (o, a, r, d) in enumerate(zip(outputs.observations.cpu(), outputs.actions.cpu(), outputs.rewards.cpu(), outputs.ends.long().cpu())):  # Make everything (N, T, ...) instead of (T, N, ...)
#             episode = Episode(o, a, r, d, torch.ones_like(d))
#             episode_id = (epoch - 1 - self.cfg.training.actor_critic.start_after_epochs) * outputs.observations.size(0) + i
#             self.episode_manager_imagination.save(episode, episode_id, epoch)

#             metrics_episode = {k: v for k, v in episode.compute_metrics().__dict__.items()}
#             metrics_episode['episode_num'] = episode_id
#             metrics_episode['action_histogram'] = wandb.Histogram(episode.actions.numpy(), num_bins=self.agent.world_model.act_vocab_size)
#             to_log.append({f'{mode_str}/{k}': v for k, v in metrics_episode.items()})

#         return to_log

#     def _save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
#         torch.save(self.agent.state_dict(), self.ckpt_dir / 'last.pt')
#         if not save_agent_only:
#             torch.save(epoch, self.ckpt_dir / 'epoch.pt')
#             torch.save({
#                 "optimizer_tokenizer": self.optimizer_tokenizer.state_dict(),
#                 "optimizer_world_model": self.optimizer_world_model.state_dict(),
#                 "optimizer_actor_critic": self.optimizer_actor_critic.state_dict(),
#             }, self.ckpt_dir / 'optimizer.pt')
#             ckpt_dataset_dir = self.ckpt_dir / 'dataset'
#             ckpt_dataset_dir.mkdir(exist_ok=True, parents=False)
#             self.train_dataset.update_disk_checkpoint(ckpt_dataset_dir)
#             if self.cfg.evaluation.should:
#                 torch.save(self.test_dataset.num_seen_episodes, self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')

#     def save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
#         tmp_checkpoint_dir = Path('checkpoints_tmp')
#         shutil.copytree(src=self.ckpt_dir, dst=tmp_checkpoint_dir, ignore=shutil.ignore_patterns('dataset'))
#         self._save_checkpoint(epoch, save_agent_only)
#         shutil.rmtree(tmp_checkpoint_dir)

#     def load_checkpoint(self) -> None:
#         assert self.ckpt_dir.is_dir()
#         self.start_epoch = torch.load(self.ckpt_dir / 'epoch.pt') + 1
#         self.agent.load(self.ckpt_dir / 'last.pt', device=self.device)
#         ckpt_opt = torch.load(self.ckpt_dir / 'optimizer.pt', map_location=self.device)
#         self.optimizer_tokenizer.load_state_dict(ckpt_opt['optimizer_tokenizer'])
#         self.optimizer_world_model.load_state_dict(ckpt_opt['optimizer_world_model'])
#         self.optimizer_actor_critic.load_state_dict(ckpt_opt['optimizer_actor_critic'])
#         self.train_dataset.load_disk_checkpoint(self.ckpt_dir / 'dataset')
#         if self.cfg.evaluation.should:
#             self.test_dataset.num_seen_episodes = torch.load(self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')
#         print(f'Successfully loaded model, optimizer and {len(self.train_dataset)} episodes from {self.ckpt_dir.absolute()}.')

#     def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
#         return {k: batch[k].to(self.device) for k in batch}

#     def finish(self) -> None:
#         wandb.finish()






















from collections import defaultdict
from functools import partial
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, Optional, Tuple

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import transformers as t
from agent import Agent
from collector import Collector
from envs import SingleProcessEnv, MultiProcessEnv
from episode import Episode
from make_reconstructions import make_reconstructions_from_batch
from models.actor_critic import ActorCritic
from models.world_model import WorldModel
from utils import configure_optimizer, EpisodeDirManager, set_seed


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
            resume=True,
            **cfg.wandb
        )

        if cfg.common.seed is not None:
            set_seed(cfg.common.seed)

        self.cfg = cfg
        self.start_epoch = 1
        self.device = torch.device(cfg.common.device)

        self.ckpt_dir = Path('checkpoints')
        self.media_dir = Path('media')
        self.episode_dir = self.media_dir / 'episodes'
        self.reconstructions_dir = self.media_dir / 'reconstructions'

        if not cfg.common.resume:
            config_dir = Path('config')
            config_path = config_dir / 'trainer.yaml'
            config_dir.mkdir(exist_ok=False, parents=False)
            shutil.copy('.hydra/config.yaml', config_path)
            wandb.save(str(config_path))
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "src"), dst="./src")
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "scripts"), dst="./scripts")
            self.ckpt_dir.mkdir(exist_ok=False, parents=False)
            self.media_dir.mkdir(exist_ok=False, parents=False)
            self.episode_dir.mkdir(exist_ok=False, parents=False)
            self.reconstructions_dir.mkdir(exist_ok=False, parents=False)

        episode_manager_train = EpisodeDirManager(self.episode_dir / 'train', max_num_episodes=cfg.collection.train.num_episodes_to_save)
        episode_manager_test = EpisodeDirManager(self.episode_dir / 'test', max_num_episodes=cfg.collection.test.num_episodes_to_save)
        self.episode_manager_imagination = EpisodeDirManager(self.episode_dir / 'imagination', max_num_episodes=cfg.evaluation.actor_critic.num_episodes_to_save)

        def create_env(cfg_env, num_envs):
            env_fn = partial(instantiate, config=cfg_env)
            return MultiProcessEnv(env_fn, num_envs, should_wait_num_envs_ratio=1.0) if num_envs > 1 else SingleProcessEnv(env_fn)

        if self.cfg.training.should:
            train_env = create_env(cfg.env.train, cfg.collection.train.num_envs)
            self.train_dataset = instantiate(cfg.datasets.train)
            self.train_collector = Collector(train_env, self.train_dataset, episode_manager_train)

        if self.cfg.evaluation.should:
            test_env = create_env(cfg.env.test, cfg.collection.test.num_envs)
            self.test_dataset = instantiate(cfg.datasets.test)
            self.test_collector = Collector(test_env, self.test_dataset, episode_manager_test)

        assert self.cfg.training.should or self.cfg.evaluation.should
        env = train_env if self.cfg.training.should else test_env

        tokenizer = instantiate(cfg.tokenizer)
        world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=env.num_actions, config=instantiate(cfg.world_model))
        actor_critic = ActorCritic(**cfg.actor_critic, act_vocab_size=env.num_actions)
        self.agent = Agent(tokenizer, world_model, actor_critic).to(self.device)
        print(f'{sum(p.numel() for p in self.agent.tokenizer.parameters())} parameters in agent.tokenizer')
        print(f'{sum(p.numel() for p in self.agent.world_model.parameters())} parameters in agent.world_model')
        print(f'{sum(p.numel() for p in self.agent.actor_critic.parameters())} parameters in agent.actor_critic')
        self.model = t.IrisModel(t.IrisConfig())
        self.model.to(self.device)
        self.optimizer_tokenizer = torch.optim.Adam(self.model.rl_agent.discrete_autoencoder.parameters(), lr=cfg.training.learning_rate)
        self.optimizer_world_model = configure_optimizer(self.model.rl_agent.world_model, cfg.training.learning_rate, cfg.training.world_model.weight_decay)
        self.optimizer_actor_critic = torch.optim.Adam(self.model.rl_agent.actor_critic.parameters(), lr=cfg.training.learning_rate)

        # if cfg.initialization.path_to_checkpoint is not None:
        #     self.agent.load(**cfg.initialization, device=self.device)

        # if cfg.common.resume:
        #     self.load_checkpoint()

    def run(self) -> None:

        for epoch in range(self.start_epoch, 1 + self.cfg.common.epochs):

            print(f"\nEpoch {epoch} / {self.cfg.common.epochs}\n")
            start_time = time.time()
            to_log = []

            if self.cfg.training.should:
                if epoch <= self.cfg.collection.train.stop_after_epochs:
                    to_log += self.train_collector.collect(self.agent, epoch, **self.cfg.collection.train.config)
                    # print("ttttttttt", to_log)
                to_log += self.train_agent(epoch)

            # if self.cfg.evaluation.should and (epoch % self.cfg.evaluation.every == 0):
            #     self.test_dataset.clear()
            #     to_log += self.test_collector.collect(self.agent, epoch, **self.cfg.collection.test.config)
            #     to_log += self.eval_agent(epoch)

            if self.cfg.training.should:
                self.save_checkpoint(epoch, save_agent_only=not self.cfg.common.do_checkpoint)

            to_log.append({'duration': (time.time() - start_time) / 3600})
            for metrics in to_log:
                wandb.log({'epoch': epoch, **metrics})

        self.finish()

    def train_agent(self, epoch: int) -> None:
        self.model.train()
        # self.agent.zero_grad()

        metrics_tokenizer, metrics_world_model, metrics_actor_critic = {}, {}, {}

        cfg_tokenizer = self.cfg.training.tokenizer
        cfg_world_model = self.cfg.training.world_model
        cfg_actor_critic = self.cfg.training.actor_critic

        
        
        if epoch > cfg_tokenizer.start_after_epochs:
            
            
            metrics_tokenizer = self.train_component('tokenizer',0, self.optimizer_tokenizer, **cfg_tokenizer)
        self.model.rl_agent.discrete_autoencoder.eval()

        if epoch > cfg_world_model.start_after_epochs:
            metrics_world_model = self.train_component('world_model',1, self.optimizer_world_model, **cfg_world_model)
        self.model.rl_agent.world_model.eval()

        if epoch > cfg_actor_critic.start_after_epochs:
            metrics_actor_critic = self.train_component('actor_Critic',2, self.optimizer_actor_critic, **cfg_actor_critic)
        self.model.rl_agent.actor_critic.eval()

        print("metricssssssssS", **metrics_tokenizer, **metrics_world_model, **metrics_actor_critic)

        return [{'epoch': epoch, **metrics_tokenizer, **metrics_world_model, **metrics_actor_critic}]

    def train_component(self, component:str, idx, optimizer: torch.optim.Optimizer, steps_per_epoch: int, grad_acc_steps: int, max_grad_norm: Optional[float], **kwargs_loss: Any) -> Dict[str, float]:
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)

        for _ in tqdm(range(steps_per_epoch), desc=f"Training {component}", file=sys.stdout):
            optimizer.zero_grad()
            for _ in range(grad_acc_steps):
                batch_t = self.train_dataset.sample_batch(self.cfg.training.tokenizer.batch_num_samples, 1, True)
                batch_t = self._to_device(batch_t)
                batch_wm = self.train_dataset.sample_batch(self.cfg.training.world_model.batch_num_samples, self.cfg.common.sequence_length, True)
                batch_wm = self._to_device(batch_wm)
                batch_ac = self.train_dataset.sample_batch(self.cfg.training.actor_critic.batch_num_samples, 1 + self.cfg.training.actor_critic.burn_in, False)
                batch_ac = self._to_device(batch_ac)
                
                outputs = self.model(observations = [batch_t['observations'],batch_wm['observations'],batch_ac['observations']],
                                            actions = [batch_t['actions'],batch_wm['actions'],batch_ac['actions']],
                                            rewards = [batch_t['rewards'],batch_wm['rewards'],batch_ac['rewards']],
                                            ends = [batch_t['ends'],batch_wm['ends'],batch_ac['ends']],
                                            mask_padding = [batch_t['mask_padding'],batch_wm['mask_padding'],batch_ac['mask_padding']],
                                            should_preprocess = False,
                                            should_postprocess = False,
                                            output_hidden_states = True,
                                            output_attentions = False,
                                            return_dict = True)
                losses = outputs.losses[idx]
                with torch.autograd.set_detect_anomaly(True):
                    loss_total_step = losses.loss_total
                    loss_total_step.backward()
                    loss_total_epoch += loss_total_step.item() / steps_per_epoch

                    for loss_name, loss_value in losses.intermediate_losses.items():
                        intermediate_losses[f"{component}/train/{loss_name}"] += loss_value / steps_per_epoch

            if max_grad_norm is not None:
                if component == "tokenizer":
                    torch.nn.utils.clip_grad_norm_(self.model.rl_agent.discrete_autoencoder.parameters(), max_grad_norm)
                elif component == "world_model":
                    torch.nn.utils.clip_grad_norm_(self.model.rl_agent.world_model.parameters(), max_grad_norm)
                else :
                    torch.nn.utils.clip_grad_norm_(self.model.rl_agent.actor_critic.parameters(), max_grad_norm)

            optimizer.step()

        metrics = {f'{component}/train/total_loss': loss_total_epoch, **intermediate_losses}
        return metrics

    @torch.no_grad()
    def eval_agent(self, epoch: int) -> None:
        self.agent.eval()

        metrics_tokenizer, metrics_world_model = {}, {}

        cfg_tokenizer = self.cfg.evaluation.tokenizer
        cfg_world_model = self.cfg.evaluation.world_model
        cfg_actor_critic = self.cfg.evaluation.actor_critic

        if epoch > cfg_tokenizer.start_after_epochs:
            metrics_tokenizer = self.eval_component(self.agent.tokenizer, cfg_tokenizer.batch_num_samples, sequence_length=1)

        if epoch > cfg_world_model.start_after_epochs:
            metrics_world_model = self.eval_component(self.agent.world_model, cfg_world_model.batch_num_samples, sequence_length=self.cfg.common.sequence_length, tokenizer=self.agent.tokenizer)

        if epoch > cfg_actor_critic.start_after_epochs:
            self.inspect_imagination(epoch)

        if cfg_tokenizer.save_reconstructions:
            batch = self._to_device(self.test_dataset.sample_batch(batch_num_samples=3, sequence_length=self.cfg.common.sequence_length))
            make_reconstructions_from_batch(batch, save_dir=self.reconstructions_dir, epoch=epoch, tokenizer=self.agent.tokenizer)

        return [metrics_tokenizer, metrics_world_model]

    @torch.no_grad()
    def eval_component(self, component: nn.Module, batch_num_samples: int, sequence_length: int, **kwargs_loss: Any) -> Dict[str, float]:
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)

        steps = 0
        pbar = tqdm(desc=f"Evaluating {str(component)}", file=sys.stdout)
        for batch in self.test_dataset.traverse(batch_num_samples, sequence_length):
            batch = self._to_device(batch)

            losses = component.compute_loss(batch, **kwargs_loss)
            loss_total_epoch += losses.loss_total.item()

            for loss_name, loss_value in losses.intermediate_losses.items():
                intermediate_losses[f"{str(component)}/eval/{loss_name}"] += loss_value

            steps += 1
            pbar.update(1)

        intermediate_losses = {k: v / steps for k, v in intermediate_losses.items()}
        metrics = {f'{str(component)}/eval/total_loss': loss_total_epoch / steps, **intermediate_losses}
        return metrics

    @torch.no_grad()
    def inspect_imagination(self, epoch: int) -> None:
        mode_str = 'imagination'
        batch = self.test_dataset.sample_batch(batch_num_samples=self.episode_manager_imagination.max_num_episodes, sequence_length=1 + self.cfg.training.actor_critic.burn_in, sample_from_start=False)
        outputs = self.agent.actor_critic.imagine(self._to_device(batch), self.agent.tokenizer, self.agent.world_model, horizon=self.cfg.evaluation.actor_critic.horizon, show_pbar=True)

        to_log = []
        for i, (o, a, r, d) in enumerate(zip(outputs.observations.cpu(), outputs.actions.cpu(), outputs.rewards.cpu(), outputs.ends.long().cpu())):  # Make everything (N, T, ...) instead of (T, N, ...)
            episode = Episode(o, a, r, d, torch.ones_like(d))
            episode_id = (epoch - 1 - self.cfg.training.actor_critic.start_after_epochs) * outputs.observations.size(0) + i
            self.episode_manager_imagination.save(episode, episode_id, epoch)

            metrics_episode = {k: v for k, v in episode.compute_metrics().__dict__.items()}
            metrics_episode['episode_num'] = episode_id
            metrics_episode['action_histogram'] = wandb.Histogram(episode.actions.numpy(), num_bins=self.agent.world_model.act_vocab_size)
            to_log.append({f'{mode_str}/{k}': v for k, v in metrics_episode.items()})

        return to_log

    def _save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        torch.save(self.agent.state_dict(), self.ckpt_dir / 'last.pt')
        if not save_agent_only:
            torch.save(epoch, self.ckpt_dir / 'epoch.pt')
            torch.save({
                "optimizer_tokenizer": self.optimizer_tokenizer.state_dict(),
                "optimizer_world_model": self.optimizer_world_model.state_dict(),
                "optimizer_actor_critic": self.optimizer_actor_critic.state_dict(),
            }, self.ckpt_dir / 'optimizer.pt')
            ckpt_dataset_dir = self.ckpt_dir / 'dataset'
            ckpt_dataset_dir.mkdir(exist_ok=True, parents=False)
            self.train_dataset.update_disk_checkpoint(ckpt_dataset_dir)
            if self.cfg.evaluation.should:
                torch.save(self.test_dataset.num_seen_episodes, self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')

    def save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        tmp_checkpoint_dir = Path('checkpoints_tmp')
        shutil.copytree(src=self.ckpt_dir, dst=tmp_checkpoint_dir, ignore=shutil.ignore_patterns('dataset'))
        self._save_checkpoint(epoch, save_agent_only)
        shutil.rmtree(tmp_checkpoint_dir)

    def load_checkpoint(self) -> None:
        assert self.ckpt_dir.is_dir()
        self.start_epoch = torch.load(self.ckpt_dir / 'epoch.pt') + 1
        self.agent.load(self.ckpt_dir / 'last.pt', device=self.device)
        ckpt_opt = torch.load(self.ckpt_dir / 'optimizer.pt', map_location=self.device)
        self.optimizer_tokenizer.load_state_dict(ckpt_opt['optimizer_tokenizer'])
        self.optimizer_world_model.load_state_dict(ckpt_opt['optimizer_world_model'])
        self.optimizer_actor_critic.load_state_dict(ckpt_opt['optimizer_actor_critic'])
        self.train_dataset.load_disk_checkpoint(self.ckpt_dir / 'dataset')
        if self.cfg.evaluation.should:
            self.test_dataset.num_seen_episodes = torch.load(self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')
        print(f'Successfully loaded model, optimizer and {len(self.train_dataset)} episodes from {self.ckpt_dir.absolute()}.')

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.device) for k in batch}

    def finish(self) -> None:
        wandb.finish()