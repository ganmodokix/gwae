import itertools
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from vaetc.network.cnn import ConvGaussianEncoder, ConvDecoder
from vaetc.models.utils import detach_dict
from vaetc.models.vae import VAE
from vaetc.models.infovae import MMDVAE, mmd
from vaetc.data.utils import IMAGE_SHAPE

from .causal import DifferentiableDAG, CausalLinear

BATCHNORM_MOMENTUM = 0.5

class Discriminator(nn.Module):

    def __init__(self, x_shape: tuple[int, int, int], z_dim: int) -> None:
        super().__init__()

        self.x_net = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 8, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(8, 16, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(16, 32, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            
            spectral_norm((nn.Conv2d(32, 64, 4, 2, 1))),
            nn.LeakyReLU(0.2),
            
            spectral_norm((nn.Conv2d(64, 128, 4, 2, 1))),
            nn.LeakyReLU(0.2),
            
            spectral_norm((nn.Conv2d(128, 256, 4, 2, 1))),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            spectral_norm(nn.Linear(256 * 1 * 1, 64)),

        )

        self.z_net = nn.Sequential(
            spectral_norm(nn.Linear(z_dim, 256)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Linear(256, 256)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Linear(256, 64)),
            nn.LeakyReLU(0.2),
        )

        self.xz_net = nn.Sequential(
            spectral_norm(nn.Linear(128, 256)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Linear(256, 256)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Linear(256, 1)),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:

        h = torch.cat([self.x_net(x), self.z_net(z)], dim=1) / 2
        return self.xz_net(h)

class LeakySiLU(nn.Module):
    """ x * sigmoid(x) + ax * (1-sigmoid(x)) """

    def __init__(self, negative_slope=0.01) -> None:
        super().__init__()

        self.negative_slope = float(negative_slope)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        return (1 - self.negative_slope) * F.silu(input) + self.negative_slope * input

class ResidualMinusTanh(nn.Module):
    """ x - a * tanh(x) """

    def __init__(self) -> None:
        super().__init__()

        self.logit_center_slope = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        center_slope = torch.tanh(self.logit_center_slope)
        return input - center_slope * torch.tanh(input)

class Sampler(nn.Module):

    def __init__(self, z_dim: int):
        super().__init__()

        self.z_dim = int(z_dim)
        assert self.z_dim > 0

    def forward(self, batch_size: int): # no input

        raise NotImplementedError("return (batch_size, z_dim)-shaped tensor")

class FactorizedNormalSampler(Sampler):

    def __init__(self, z_dim: int):
        super().__init__(z_dim)

    def forward(self, batch_size: int):
        
        return torch.randn(size=[batch_size, self.z_dim]).cuda()

class FactorizedUniformSampler(Sampler):

    def __init__(self, z_dim: int):
        super().__init__(z_dim)

    def forward(self, batch_size: int):
        
        u = torch.rand(size=[batch_size, self.z_dim]).cuda()
        u = u * 2 - 1
        u = u * 3 ** 0.5
        return u

class SphericalUniformSampler(Sampler):

    def __init__(self, z_dim: int):
        super().__init__(z_dim)

    def forward(self, batch_size: int):
        
        r = torch.randn(size=[batch_size, self.z_dim], device="cuda")
        return F.normalize(r)

class NeuralSampler(Sampler):
    """
    Sample z~π_θ(z|ε) factorized, where ε~N(0,I), E[z_i]=0, V[z_i]=I for all i
    """

    def __init__(self, z_dim: int):
        super().__init__(z_dim)

        hidden_dim = self.z_dim * 4
        self.net = nn.Sequential(
            nn.Linear(self.z_dim, hidden_dim),
            LeakySiLU(negative_slope=0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            LeakySiLU(negative_slope=0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            LeakySiLU(negative_slope=0.2),
            
            nn.Linear(hidden_dim, self.z_dim),
            nn.BatchNorm1d(self.z_dim, momentum=BATCHNORM_MOMENTUM, affine=False),
        )
        
    def forward(self, batch_size: int):
        
        eps = torch.randn(size=[batch_size, self.z_dim]).cuda()
        z = self.net(eps)
        return z

class FactorizedNeuralSampler(Sampler):

    def __init__(self, z_dim: int):
        super().__init__(z_dim)

        hidden_ratio = 4
        self.net = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=[self.z_dim, 1]),
            nn.Conv1d(self.z_dim, self.z_dim * hidden_ratio, 1, groups=self.z_dim),
            LeakySiLU(negative_slope=0.2),
            nn.Conv1d(self.z_dim * hidden_ratio, self.z_dim * hidden_ratio, 1, groups=self.z_dim),
            LeakySiLU(negative_slope=0.2),
            nn.Conv1d(self.z_dim * hidden_ratio, self.z_dim * hidden_ratio, 1, groups=self.z_dim),
            LeakySiLU(negative_slope=0.2),
            nn.Conv1d(self.z_dim * hidden_ratio, self.z_dim, 1, groups=self.z_dim),
            nn.BatchNorm1d(self.z_dim, momentum=BATCHNORM_MOMENTUM, affine=False),
            nn.Flatten(),
        )
        
    def forward(self, batch_size: int):
        
        eps = torch.randn(size=[batch_size, self.z_dim]).cuda()
        z = self.net(eps)
        return z


class CausalSampler(Sampler):

    def __init__(self, z_dim: int):
        super().__init__(z_dim)

        self.dag = DifferentiableDAG(self.z_dim)
        self.ccs = nn.ModuleList([
            CausalLinear(self.z_dim),
            CausalLinear(self.z_dim),
            CausalLinear(self.z_dim),
            CausalLinear(self.z_dim),
            CausalLinear(self.z_dim),
            CausalLinear(self.z_dim),
            CausalLinear(self.z_dim),
        ])

        self.batchnorm = nn.BatchNorm1d(self.z_dim, momentum=BATCHNORM_MOMENTUM)

    def forward(self, batch_size: int):
        
        eps = torch.randn(size=[batch_size, self.z_dim]).cuda()

        adj_mask = self.dag()
        h = eps
        for cc in self.ccs:
            h = cc(eps, adj_mask)

        return self.batchnorm(h)

class GaussianMixtureSampler(Sampler):

    def __init__(self, z_dim: int, num_components: int):
        super().__init__(z_dim)

        self.num_components = int(num_components)
        assert self.num_components >= 2
        
        self.component_mean = nn.Parameter(torch.randn(size=[self.num_components, self.z_dim]), requires_grad=True)

        eyes = torch.eye(self.z_dim)[None,:,:].tile(self.num_components, 1, 1)
        noises = torch.randn(size=[self.num_components, self.z_dim, self.z_dim])
        csqrtp = eyes + noises * 0.01
        self.component_sqrtprecision = nn.Parameter(csqrtp, requires_grad=True)

        self.logits_unnormalized = nn.Parameter(torch.randn(size=[self.num_components, ]), requires_grad=True)

        self.batchnorm = nn.BatchNorm1d(self.z_dim, momentum=BATCHNORM_MOMENTUM, affine=False)

    def forward(self, batch_size: int) -> torch.Tensor:

        logits = self.logits_unnormalized - torch.logsumexp(self.logits_unnormalized, dim=0)

        u = logits[None,:].tile(batch_size, 1)
        g = u - torch.log(-torch.log(torch.rand_like(u)))
        membership = torch.argmax(g, dim=1)

        eps = torch.randn(size=[batch_size, self.z_dim], device=logits.device)
        z = self.component_mean[membership] + (eps[:,None,:] * self.component_sqrtprecision[membership,:,:]).sum(dim=1)
        z = self.batchnorm(z)

        return z

class VectorQuantizedSampler(Sampler):

    def __init__(self, z_dim: int, num_embeddings: int):
        super().__init__(z_dim)

        self.num_embeddings = int(num_embeddings)
        assert self.num_embeddings > 0

        self.embeddings = nn.Parameter(torch.randn(size=[self.num_embeddings, self.z_dim]), requires_grad=True)

    def forward(self, batch_size: int) -> torch.Tensor:

        z = torch.randn(size=[batch_size, self.z_dim], device=self.embeddings.device)
        dist = ((z[:,None,:] - self.embeddings[None,:,:]) ** 2).sum(dim=2)
        z_embs = self.embeddings[dist.argmin(dim=1)]
        return z_embs

def build_sampler(sampler_type: str, z_dim: int, hyperparameters: dict) -> Sampler:

        if sampler_type == "neural":
            sampler = NeuralSampler(z_dim)
        elif sampler_type == "causal":
            sampler = CausalSampler(z_dim)
        elif sampler_type == "factorized_normal":
            sampler = FactorizedNormalSampler(z_dim)
        elif sampler_type == "factorized_uniform":
            sampler = FactorizedUniformSampler(z_dim)
        elif sampler_type == "factorized_neural":
            sampler = FactorizedNeuralSampler(z_dim)
        elif sampler_type == "spherical":
            sampler = SphericalUniformSampler(z_dim)
        elif sampler_type == "gm":
            num_components = int(hyperparameters["num_components"])
            sampler = GaussianMixtureSampler(z_dim, num_components)
        elif sampler_type == "vq":
            num_embeddings = int(hyperparameters["num_embeddings"])
            sampler = VectorQuantizedSampler(z_dim, num_embeddings)
        else:
            raise RuntimeError(f"Invalid sampler type '{sampler_type}'")

        return sampler

def logit_gradient(x: torch.Tensor, z: torch.Tensor, logit: torch.Tensor):

    batch_size = x.shape[0]

    grad = torch.autograd.grad(
        outputs=logit,
        inputs=[x, z],
        grad_outputs=torch.ones_like(logit),
        retain_graph=True,
        create_graph=True,
    )
    grad_x = grad[0].view(batch_size, -1)
    grad_z = grad[1].view(batch_size, -1)
    grad_cat = torch.cat([grad_x, grad_z], dim=1)

    return grad_cat

def gradient_penalty_one_centered(
    x: torch.Tensor, z: torch.Tensor,
    x2s: torch.Tensor, zs: torch.Tensor,
    disc_block: nn.Module
) -> torch.Tensor:

    batch_size = x.shape[0]
    
    tp = torch.rand(size=[batch_size, ], device="cuda")
    xp = x * tp[:,None,None,None] + x2s * (1 - tp[:,None,None,None])
    zp = z * tp[:,None] + zs * (1 - tp[:,None])
    
    xp.requires_grad_(True)
    zp.requires_grad_(True)

    logit_interpolation = disc_block(xp, zp)
    grad_cat = logit_gradient(xp, zp, logit_interpolation)
    grad_norm = grad_cat.norm(p=2, dim=1)
    loss_gp = torch.mean((grad_norm - 1) ** 2)

    return loss_gp

def gradient_penalty_zero_centered(
    x: torch.Tensor, z: torch.Tensor,
    logit_autoencoding: torch.Tensor,
    x2s: torch.Tensor, zs: torch.Tensor,
    logit_sampling: torch.Tensor,
) -> torch.Tensor:

    autoencoding_grad_cat = logit_gradient(x, z, logit_autoencoding)
    autoencoding_grad_d2 = autoencoding_grad_cat.square().sum(dim=1)

    sampling_grad_cat = logit_gradient(x2s, zs, logit_sampling)
    sampling_grad_d2 = sampling_grad_cat.square().sum(dim=1)

    return autoencoding_grad_d2.mean() * 0.5 + sampling_grad_d2.mean() * 0.5

class ConvGaussianEncoderBN(ConvGaussianEncoder):

    def __init__(self, z_dim: int, in_features: int = 3, batchnorm_momentum: float = 0.1, inplace: bool = True, resblock: bool = False):

        super().__init__(z_dim, in_features, False, batchnorm_momentum, inplace, resblock)
        
        self.momentum = float(batchnorm_momentum)
        self.running_mean = nn.Parameter(torch.randn(size=[z_dim]), requires_grad=False)
        self.running_logvar = nn.Parameter(torch.randn(size=[z_dim]), requires_grad=False)
        self.num_batches_tracked = nn.Parameter(torch.tensor(0), requires_grad=False)

    def forward(self, x):

        mean, logvar = super().forward(x)
        mean: torch.Tensor
        logvar: torch.Tensor

        if self.training:

            mean_batch = mean.mean(dim=0)
            logvar_batch = -math.log(logvar.shape[0]) + logvar.logsumexp(dim=0)

            exmean = mean_batch
            exlogvar = torch.stack([logvar_batch, torch.clamp(mean.var(dim=0), min=1e-10).log()], dim=0).logsumexp(dim=0)
            
            if self.num_batches_tracked.item() == 0:
                self.running_mean.copy_(exmean.detach())
                self.running_logvar.copy_(exlogvar.detach())
            else:
                self.running_mean.copy_((exmean * (1. - self.momentum) + self.running_mean * self.momentum).detach())
                self.running_logvar.copy_((exlogvar * (1. - self.momentum) + self.running_logvar * self.momentum).detach())
        
            self.num_batches_tracked.copy_((self.num_batches_tracked + 1).detach())
        
        else:

            exmean = self.running_mean
            exlogvar = self.running_logvar

        mean = (mean - exmean[None,:]) * (-0.5 * exlogvar).exp()[None,:]
        logvar = logvar - exlogvar[None,:]

        return mean, logvar

class GromovWassersteinAutoEncoder(VAE):

    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)

        self.sampler_type = str(hyperparameters["sampler_type"])
        self.sampler = build_sampler(self.sampler_type, self.z_dim, hyperparameters)

        batchnorm = hyperparameters.get("batchnorm", False)
        resblock = hyperparameters.get("resblock", False)
        batchnorm_momentum = BATCHNORM_MOMENTUM
        latent_bn = bool(hyperparameters.get("latent_bn", False))
        if latent_bn:
            self.enc_block = ConvGaussianEncoderBN(self.z_dim, batchnorm_momentum=batchnorm_momentum, resblock=resblock)
        else:
            self.enc_block = ConvGaussianEncoder(self.z_dim, batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum, resblock=resblock)
        self.dec_block = ConvDecoder(self.z_dim, batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum, resblock=resblock)

        self.coef_w = float(hyperparameters["coef_w"])
        self.coef_d = float(hyperparameters["coef_d"])
        self.coef_qentropy = float(hyperparameters["coef_qentropy"])
        self.merged_condition = bool(hyperparameters["merged_condition"])
        self.learned_similarity = bool(hyperparameters.get("learned_similarity", False))
        assert (not self.learned_similarity) or self.merged_condition
        self.mixed_potential = bool(hyperparameters.get("mixed_potential", False))

        if self.merged_condition:
            
            self.disc_block = Discriminator(IMAGE_SHAPE, self.z_dim)

            self.lr_disc = float(hyperparameters["lr_disc"])
            self.coef_gp = float(hyperparameters["coef_gp"])
            self.times_d_training = int(hyperparameters["times_d_training"])

        self.distance_coef_initial = float(hyperparameters.get("distance_coef_initial", 1.0))
        self.log_distance_coef = nn.Parameter(torch.tensor(self.distance_coef_initial).log(), requires_grad=True)

    def build_optimizers(self) -> dict[str, torch.optim.Optimizer]:

        main_parameters = itertools.chain(
            self.enc_block.parameters(),
            self.dec_block.parameters(),
            self.sampler.parameters(),
            [self.log_distance_coef],
        )
        
        if self.merged_condition:
            disc_parameters = self.disc_block.parameters()
            return {
                "main": torch.optim.RMSprop(main_parameters, lr=self.lr),
                "disc": torch.optim.RMSprop(disc_parameters, lr=self.lr_disc),
            }
        else:
            return {
                "main": torch.optim.Adam(main_parameters, lr=self.lr, betas=(0.9, 0.999)),
            }

    def sample_prior(self, batch_size: int):
        
        zs = self.sampler(batch_size)

        return zs

    def encode_gauss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = self.enc_block(x)
        return mean, logvar

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        mean, logvar = self.encode_gauss(x)
        z = self.reparameterize(mean, logvar)
        x2 = self.dec_block(z)
        zs = self.sample_prior(z.shape[0])
        xs = self.decode(zs)

        if self.merged_condition:

            logit_fake = self.disc_block(x, z)
            logit_true = self.disc_block(xs, zs)

            return mean, logvar, z, x2, logit_fake, zs, logit_true

        else:

            return mean, logvar, z, x2, zs

    def step_batch(self, batch: tuple[torch.Tensor, torch.Tensor], optimizers=None, progress=None, training=False):
        
        x, t = batch
        x = x.cuda()
        batch_size = x.shape[0]

        # ========================================
        #  Main training
        # ========================================

        mean, logvar = self.enc_block(x)
        z = self.reparameterize(mean, logvar)
        x2 = self.dec_block(z)

        zs = self.sample_prior(batch_size)
        x2s = self.dec_block(zs)

        # scale_z = zs.std()
        scale_z = zs.std(dim=0).mean()
        scale_z = torch.clamp(scale_z, min=1e-8)

        dx_elem = (x2s[:,None,...] - x2s[None,:,...]).view(batch_size, batch_size, -1)
        dz_elem = (zs[:,None,:] - zs[None,:,:]) * (self.log_distance_coef.exp() / scale_z)
        dx = dx_elem.norm(p=2, dim=2)
        dz = dz_elem.norm(p=2, dim=2)

        # GW(data distro, model prior)
        loss_gw = (torch.abs(dz - dx)).mean()

        # W(data, model)
        pixel_error = (0.5 * (x - x2) ** 2).view(batch_size, -1).sum(dim=1).mean()
        if self.learned_similarity:
            learned_error = (0.5 * (self.disc_block.x_net(x) - self.disc_block.x_net(x2)) ** 2).sum(dim=1).mean()
            loss_w = pixel_error + learned_error
        else:
            loss_w = pixel_error
        loss_qentropy = -logvar.sum(dim=1).mean()

        # W(inference, generation)
        if self.merged_condition:

            logit_autoencoding = self.disc_block(x, z)
            if self.mixed_potential:
                zq = z.detach()
                xq = self.dec_block(zq)
                logit_sampling = (self.disc_block(x2s, zs) + self.disc_block(xq, zq)) / 2
            else:
                logit_sampling = self.disc_block(x2s, zs)
            loss_d = (logit_sampling - logit_autoencoding).mean()

        else:

            loss_d = mmd(z, zs)

        loss = loss_gw + loss_w * self.coef_w + loss_qentropy * self.coef_qentropy + loss_d * self.coef_d
        
        if training:
            self.zero_grad()
            loss.backward()
            optimizers["main"].step()

        # ========================================
        #  Disc. training
        # ========================================

        if self.merged_condition:

            mean, logvar = self.enc_block(x)

            for _ in range(self.times_d_training if training else 1):

                z = self.reparameterize(mean, logvar)
                x2 = self.dec_block(z)
                zs = self.sample_prior(batch_size)
                x2s = self.dec_block(zs)

                z = z.detach()
                x2 = x2.detach()
                zs = zs.detach()
                x2s = x2s.detach()

                logit_autoencoding = self.disc_block(x, z) # ↓
                if self.mixed_potential:
                    logit_sampling = (self.disc_block(x2s, zs) + self.disc_block(x2, z)) / 2 # ↑
                else:
                    logit_sampling = self.disc_block(x2s, zs) # ↑

                loss_logits = -(logit_sampling - logit_autoencoding).mean()

                loss_gp = gradient_penalty_one_centered(x, z, x2s, zs, self.disc_block) \
                            + 1e-4 * logit_autoencoding.square().mean() \
                            + 1e-4 * logit_sampling.square().mean()

                # x.requires_grad_(True)
                # z.requires_grad_(True)
                # x2s.requires_grad_(True)
                # zs.requires_grad_(True)
                # loss_gp = gradient_penalty_zero_centered(x, z, logit_autoencoding, x2s, zs, logit_sampling)

                loss_disc = loss_logits + loss_gp * self.coef_gp

                if training:
                    self.zero_grad()
                    loss_disc.backward()
                    optimizers["disc"].step()

        if self.merged_condition:

            return detach_dict({
                
                "loss": loss,

                "loss_gw": loss_gw,
                "loss_w": loss_w,
                "loss_d": loss_d,
                "loss_qentropy": loss_qentropy,
                
                "loss_disc": loss_disc,
                "loss_logits": loss_logits,
                "loss_gp": loss_gp,
                "logit_autoencoding": logit_autoencoding.mean(),
                "logit_sampling": logit_sampling.mean(),

                "distance_coef": self.log_distance_coef.exp(),
                "qz_scale": z.std(dim=0).mean(),
                "pz_scale": zs.std(dim=0).mean(),
            })

        else:

            return detach_dict({
                
                "loss": loss,

                "loss_gw": loss_gw,
                "loss_w": loss_w,
                "loss_d": loss_d,
                "loss_qentropy": loss_qentropy,

                "distance_coef": self.log_distance_coef.exp(),
                "qz_scale": z.std(dim=0).mean(),
                "pz_scale": zs.std(dim=0).mean(),

            })

    def train_batch(self, batch, optimizers, progress: float):
        return self.step_batch(batch, optimizers=optimizers, progress=progress, training=True)

    def eval_batch(self, batch):
        return self.step_batch(batch, training=False)
