import torch
from .tools import MonitorRL

class MonitorHnet(MonitorRL):
    def __init__(self, hparams, agent, mnet, hnet, collector):
        super(MonitorHnet, self).__init__(hparams, agent, mnet, collector, None)
        self.mnet = mnet
        self.hnet = hnet
        self.model_to_save = {'mnet': mnet, 'hnet': hnet}

        self.loss_task = 0
        self.loss_reg = 0

    def train_step(self, loss_task, loss_reg, dTheta, grad_tloss, weights):
        self.loss_task += loss_task.item()
        self.loss_reg += loss_reg.item()
        if (self.train_iter % self.print_train_every == 0):
            self.loss_task /= self.print_train_every
            self.loss_reg /= self.print_train_every
            loss_tot = self.loss_reg + self.loss_task
            print(f"Batch: {self.train_iter}, Loss: {loss_tot:.5f}, " + 
                  f"Task L: {self.loss_task:.5f}, Reg L: {self.loss_reg:.5f}")

            i = self.train_iter

            self.writer.add_scalar('train/loss', self.loss_task, i)
            self.writer.add_scalar('train/regularizer', self.loss_reg, i)
            self.writer.add_scalar('train/total_loss', loss_tot, i)
            if dTheta is not None:
                dT_norm = torch.norm(torch.cat([d.view(-1) for d in dTheta]), 2)
                self.writer.add_scalar('train/delta_theta_norm', dT_norm, i)
            if grad_tloss is not None:
                (grad_tloss, grad_full, grad_diff_norm, grad_cos) = grad_tloss
                self.writer.add_scalar('train/gradient_norm',
                                  torch.norm(grad_full, 2), i)
                self.writer.add_scalar('train/regularizer_gradient_norm',
                                  grad_diff_norm, i)
                self.writer.add_scalar('train/gradient_cosine_similarity',
                                  grad_cos, i)
            
            self.loss_task = 0
            self.loss_reg = 0

        if (self.train_iter % self.log_hist_every == 0):
            for i, weight in enumerate(weights):
                self.writer.add_histogram(f'train/weights/{i}', weight.flatten(), self.train_iter)
        self.train_iter += 1

    def data_aggregate_step(self, x_tt, task_id, it):
        if self.hparams.env == "lqr10":
            l2_pos = np.linalg.norm(x_tt[:10])
            l2_vel = np.linalg.norm(x_tt[10:])
            self.writer.add_scalar(f'lqr10/{task_id}/l2_pos', l2_pos, it)
            self.writer.add_scalar(f'lqr10/{task_id}/l2_vel', l2_vel, it)
        elif self.hparams.env.startswith("spaceEnv") and x_tt is not None:
            import numpy as np
            theta_margin_deg = np.degrees((x_tt[7] + 1.0) * (3 * np.pi / 4) - np.pi / 2)
            att_err_deg = 2 * np.degrees(np.arccos(np.clip(np.abs(x_tt[0]), 0.0, 1.0)))
            omega_norm_degs = np.degrees(np.linalg.norm(x_tt[4:7]) * 5.0)
            self.writer.add_scalar(f'random_phase/task_{task_id}/theta_margin_deg', theta_margin_deg, it)
            self.writer.add_scalar(f'random_phase/task_{task_id}/attitude_error_deg', att_err_deg, it)
            self.writer.add_scalar(f'random_phase/task_{task_id}/omega_norm_degs', omega_norm_degs, it)

    def validate_task(self, task_id, loader, mll, is_training=False):
        self.mnet.eval()
        self.hnet.eval()
        device = self.hparams.device
        
        # Initialize Stats
        val_loss = 0
        val_diff = 0
        N = len(loader)
        
        with torch.no_grad():
            weights = self.hnet.forward(task_id)

            for _, data in enumerate(loader):
                x_t, a_t, x_tt = data
                x_t, a_t, x_tt = x_t.to(device), a_t.to(device), x_tt.to(device)
                X = torch.cat((x_t, a_t), dim=-1)
                
                Y = self.mnet.forward(X, weights)
                
                loss = mll(Y, x_tt, weights)
                if self.hparams.out_var:
                    Y, _ = torch.split(Y, Y.size(-1)//2, dim=-1)
                diff = torch.abs(Y - x_tt).mean(dim=0)
                
                val_loss += loss
                val_diff += diff
            
            val_loss = val_loss / N
            val_diff = val_diff / N

        print(f"Iter {self.train_iter}, Task: {task_id}, " + \
              f"Val Loss: {val_loss.item():.5f}, Val Diff: {val_diff.mean().item()}")
        
        return val_loss, val_diff