from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from src.load_data import load_dataset
import config as cfg
from src.diffusion_utils import DiffusionUtils

import os
import torch


def build_diffusion(image_size, objective="pred_noise", timesteps=1024, sampling_timesteps=None,
                    is_student=False, teacher=None, using_ddim=False, use_pdistill=False, mapping_sequence=None):
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,  # 32, 64, 128
        timesteps=timesteps,    # number of steps
        sampling_timesteps=sampling_timesteps,
        objective=objective,
        is_student=is_student,
        mapping_sequence=mapping_sequence,
        teacher=teacher,
        using_ddim=using_ddim,
        ddim_sampling_eta=0.0,
        use_pdistill=use_pdistill
    ).cuda()
    return diffusion

def sampling_student(epoch=10):
    # sampling
    teacher = build_diffusion(image_size=32, timesteps=1024, objective='pred_noise', using_ddim=True)

    student = build_diffusion(image_size=32, timesteps=128, objective='pred_noise', sampling_timesteps=128,
                              is_student=True, teacher=teacher, using_ddim=True, use_pdistill=False)

    diff_util = DiffusionUtils(teacher, student_diff=student)

    d_dir = 'trained_model/diffusion_Cifar10_32x32_1024_pnoise_epoch_50.pth'
    diff_util.load_trained_model(d_dir)

    s_dir = 'saved_models/diffusion_Cifar10_32x32_128_student_pnoise_epoch_%d.pth' % epoch
    diff_util.load_trained_student(s_dir)

    num_img = 32
    shape = (num_img, student.channels, student.image_size, student.image_size)
    input_noise = torch.randn(shape, device=student.device)

    sampled_img = diff_util.sample(res_id="student", num_img=num_img, nrow=8, use_student=True, input_noise=input_noise)
    sampled_img = diff_util.sample(res_id="teacher", num_img=num_img, nrow=8, use_student=False, input_noise=input_noise)


def train_student():
    dst = 'Cifar10'
    if dst == 'Cifar10':
        teacher = build_diffusion(image_size=32, timesteps=1024, objective='pred_noise')
        student = build_diffusion(image_size=32, timesteps=128, objective='pred_noise', is_student=True, teacher=teacher)

        diff_util = DiffusionUtils(teacher, student_diff=student)

        # teacher
        d_dir = 'trained_model/diffusion_Cifar10_32x32_1024_pnoise_epoch_50.pth'

        diff_util.load_trained_model(d_dir)

        data_loader = load_dataset(batch_size=50, dataset='Cifar10', dataset_dir=cfg.DATASET_DIR)
        diff_util.set_dataloader(data_loader)

        diff_util.train_student(epochs=10, start_epochs=0, model_name="Cifar10_32x32_128_student_pnoise")

    else:
        pass


if __name__ == '__main__':
    if not os.path.exists("./sampling_res"):
        os.makedirs("./sampling_res")
    if not os.path.exists("./saved_models"):
        os.makedirs("./saved_models")

    train_student()
    sampling_student(epoch=10)

