from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from src.load_data import load_dataset
import config as cfg
from src.diffusion_utils import DiffusionUtils
from math import log10, sqrt
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

def PSNR(dummy_data, gt_data):
    '''
    PSNR metric
    '''
    mse = torch.mean((dummy_data - gt_data) ** 2).item()
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def sampling_student(epoch=10, num_img=32, input_noise=None, no_psnr=False):
    # sampling
    teacher = build_diffusion(image_size=32, timesteps=1024, objective='pred_noise', using_ddim=True)

    student = build_diffusion(image_size=32, timesteps=128, objective='pred_noise', sampling_timesteps=128,
                              is_student=True, teacher=teacher, using_ddim=True, use_pdistill=False)

    diff_util = DiffusionUtils(teacher, student_diff=student)

    d_dir = 'trained_model/diffusion_Cifar10_32x32_1024_pnoise_epoch_50.pth'
    diff_util.load_trained_model(d_dir)

    s_dir = 'saved_models/diffusion_Cifar10_32x32_128_student_pnoise_epoch_%d.pth' % epoch
    diff_util.load_trained_student(s_dir)

    if input_noise == None:
        shape = (num_img, student.channels, student.image_size, student.image_size)
        input_noise = torch.randn(shape, device=student.device)

    sampled_img_student = diff_util.sample(res_id="student_{}".format(epoch), num_img=num_img, nrow=8, use_student=True, input_noise=input_noise)
    sampled_img_teacher = diff_util.sample(res_id="teacher_{}".format(epoch), num_img=num_img, nrow=8, use_student=False, input_noise=input_noise)

    if not no_psnr:
        print("PSNR value:", PSNR(sampled_img_teacher, sampled_img_student))

def train_teacher_diffusion():
    # for training a teacher diffusion model from scratch

    dst = 'Cifar10'
    if dst == 'Bedroom':
        data_loader = load_dataset(batch_size=64, dataset='Bedroom', dataset_dir=cfg.DATASET_DIR)

        diff_util = DiffusionUtils(build_diffusion(image_size=128, timesteps=1024, objective='pred_v'))

        diff_util.set_dataloader(data_loader)
        diff_util.train(epochs=50, model_name="bedroom_1024_128x128")

    elif dst == 'Church':
        data_loader = load_dataset(batch_size=64, dataset='Church', dataset_dir=cfg.DATASET_DIR)

        diff_util = DiffusionUtils(build_diffusion(image_size=128, timesteps=1024, objective='pred_v'))

        diff_util.set_dataloader(data_loader)
        diff_util.train(epochs=50, model_name="church_1024_128x128")
    elif dst == 'Cifar10':
        data_loader = load_dataset(batch_size=50, dataset='Cifar10', dataset_dir=cfg.DATASET_DIR)

        diff_util = DiffusionUtils(build_diffusion(image_size=32, timesteps=1024, objective='pred_v'))

        diff_util.set_dataloader(data_loader)
        # d_dir = 'saved_models/diffusion_CelebA_1024_64x64_epoch_5.pth'
        # diff_util.load_trained_model(d_dir)
        diff_util.train(epochs=10, start_epochs=0, model_name="Cifar10_1024_32x32")
    elif dst == 'CelebA':
        data_loader = load_dataset(batch_size=48, dataset='CelebA', dataset_dir=cfg.DATASET_DIR)

        diff_util = DiffusionUtils(build_diffusion(image_size=64, timesteps=1024, objective='pred_noise'))

        diff_util.set_dataloader(data_loader)

        diff_util.train(epochs=30, start_epochs=0, model_name="CelebA_1024_64x64")


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
    elif dst == 'Church':

        teacher = build_diffusion(image_size=128, timesteps=1024, objective='pred_v')

        mapping_sequence = [int(i * (10.24)) for i in range(100)]

        student = build_diffusion(image_size=128, timesteps=100, objective='pred_v', is_student=True, teacher=teacher, use_pdistill=False, mapping_sequence=mapping_sequence)

        diff_util = DiffusionUtils(teacher, student_diff=student)

        d_dir = 'saved_models/church_1024_128x128_pv/diffusion_church_1024_128x128_pv_epoch_50.pth'

        diff_util.load_trained_model(d_dir)

        data_loader = load_dataset(batch_size=40, dataset='Church', dataset_dir=cfg.DATASET_DIR)

        diff_util.set_dataloader(data_loader)

        diff_util.train_student(epochs=50, start_epochs=0, model_name="Church_128x128_100_student")
    elif dst == 'CelebA':
        teacher = build_diffusion(image_size=64, timesteps=1024, objective='pred_noise')
        student = build_diffusion(image_size=64, timesteps=16, objective='pred_noise', is_student=True, teacher=teacher)

        diff_util = DiffusionUtils(teacher, student_diff=student)

        d_dir = 'saved_models/celeba_pnoise_64x64_1024/diffusion_CelebA_1024_64x64_epoch_30.pth'
        diff_util.load_trained_model(d_dir)

        data_loader = load_dataset(batch_size=25, dataset='CelebA', dataset_dir=cfg.DATASET_DIR)
        diff_util.set_dataloader(data_loader)

        diff_util.train_student(epochs=30, start_epochs=0, model_name="CelebA_64x64_16_student")
    elif dst == 'Bedroom':
        teacher = build_diffusion(image_size=128, timesteps=1024, objective='pred_v')

        student = build_diffusion(image_size=128, timesteps=100, objective='pred_v', is_student=True, teacher=teacher,
                                  use_pdistill=True, mapping_sequence=None)

        diff_util = DiffusionUtils(teacher, student_diff=student)

        d_dir = 'saved_models/Bedroom_128x128_1024_pv/diffusion_bedroom_1024_128x128_pv_epoch_50.pth'
        diff_util.load_trained_model(d_dir)

        data_loader = load_dataset(batch_size=40, dataset='Bedroom', dataset_dir=cfg.DATASET_DIR)
        diff_util.set_dataloader(data_loader)

        diff_util.train_student(epochs=50, start_epochs=0, model_name="Bedroom_128x128_100_student")


if __name__ == '__main__':
    if not os.path.exists("./sampling_res"):
        os.makedirs("./sampling_res")
    if not os.path.exists("./saved_models"):
        os.makedirs("./saved_models")

    train_student()

    num_img = 1
    shape = (num_img, 3, 32, 32)
    input_noise = torch.randn(shape).cuda()

    # show PSNR on diffirent epoch
    sampling_student(epoch=0, num_img=num_img, input_noise=input_noise)
    sampling_student(epoch=1, num_img=num_img, input_noise=input_noise)
    sampling_student(epoch=10, num_img=num_img, input_noise=input_noise)

    # show results on final epoch
    sampling_student(epoch=10, no_psnr=True)

