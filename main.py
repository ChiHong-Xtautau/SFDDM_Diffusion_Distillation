from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from src.load_data import load_dataset, visualize
import config as cfg
from src.diffusion_utils import DiffusionUtils

import torch


def build_diffusion(image_size, objective="pred_v", timesteps=1024, sampling_timesteps=None,
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


def train_ct_diffusion():
    dst = 'CelebA_HQ'
    if dst == 'Cifar10':
        data_loader = load_dataset(batch_size=50, dataset='Cifar10', dataset_dir=cfg.DATASET_DIR)
        visualize(load_dataset(batch_size=32, dataset='Cifar10', dataset_dir=cfg.DATASET_DIR),
                  save_dir='./sampling_res_new/training_data_cifar10.jpg')

        diff_util = DiffusionUtils(build_diffusion(image_size=32, timesteps=128, objective='pred_noise'))

        diff_util.set_dataloader(data_loader)
        # d_dir = 'saved_models/diffusion_ct_Cifar10_32x32_128_ct_pnoise_epoch_20.pth'
        # diff_util.load_trained_model(d_dir)
        diff_util.train_ct(epochs=100, start_epochs=20, model_name="Cifar10_32x32_128_ct_pnoise")
    elif dst == 'CelebA_HQ':
        data_loader = load_dataset(batch_size=64, dataset='CelebA_HQ', dataset_dir=cfg.DATASET_DIR)
        visualize(load_dataset(batch_size=32, dataset='CelebA_HQ', dataset_dir=cfg.DATASET_DIR),
                  save_dir='sampling_res_new/training_data_celeba_hq.jpg')

        diff_util = DiffusionUtils(build_diffusion(image_size=64, timesteps=32, objective='pred_noise'))

        diff_util.set_dataloader(data_loader)
        d_dir = 'saved_models/student/CelebA_HQ_64x64_32_pnoise_l1_sl/diffusion_CelebA_HQ_64x64_32_student_pnoise_l1_sl_epoch_35.pth'
        diff_util.load_trained_model(d_dir)
        diff_util.train_ct(epochs=35, start_epochs=30, model_name="celeba_hq_64x64_32_ct_pnoise")
    elif dst == 'Church':
        data_loader = load_dataset(batch_size=64, dataset='Church', dataset_dir=cfg.DATASET_DIR)
        visualize(load_dataset(batch_size=32, dataset='Church', dataset_dir=cfg.DATASET_DIR),
                  save_dir='sampling_res_new/training_data_church.jpg')

        diff_util = DiffusionUtils(build_diffusion(image_size=64, timesteps=128, objective='pred_noise'))

        diff_util.set_dataloader(data_loader)
        d_dir = 'saved_models/diffusion_ct_church_64x64_128_ct_pnoise_epoch_20.pth'
        diff_util.load_trained_model(d_dir)
        diff_util.train_ct(epochs=50, start_epochs=20, model_name="church_64x64_128_ct_pnoise")
    if dst == 'Bedroom':
        data_loader = load_dataset(batch_size=64, dataset='Bedroom', dataset_dir=cfg.DATASET_DIR)
        visualize(load_dataset(batch_size=32, dataset='Bedroom', dataset_dir=cfg.DATASET_DIR),
                  save_dir='./sampling_res_new/training_data_bedroom.jpg')

        diff_util = DiffusionUtils(build_diffusion(image_size=64, timesteps=128, objective='pred_noise'))

        diff_util.set_dataloader(data_loader)
        diff_util.train_ct(epochs=50, model_name="bedroom_64x64_128_ct_pnoise")


def train_diffusion():
    dst = 'CelebA_HQ'
    # dst = None
    if dst == 'CelebA':
        data_loader = load_dataset(batch_size=48, dataset='CelebA', dataset_dir=cfg.DATASET_DIR)
        visualize(load_dataset(batch_size=32, dataset='CelebA', dataset_dir=cfg.DATASET_DIR),
                  save_dir='./sampling_res/training_data_celebA.jpg')

        diff_util = DiffusionUtils(build_diffusion(image_size=64, timesteps=1024, objective='pred_noise'))

        diff_util.set_dataloader(data_loader)
        # d_dir = 'saved_models/diffusion_CelebA_1024_64x64_epoch_5.pth'
        # diff_util.load_trained_model(d_dir)
        diff_util.train(epochs=30, start_epochs=0, model_name="CelebA_1024_128x128_px0")
    if dst == 'Bedroom':
        data_loader = load_dataset(batch_size=64, dataset='Bedroom', dataset_dir=cfg.DATASET_DIR)
        visualize(load_dataset(batch_size=32, dataset='Bedroom', dataset_dir=cfg.DATASET_DIR),
                  save_dir='./sampling_res/training_data_bedroom.jpg')

        diff_util = DiffusionUtils(build_diffusion(image_size=64, timesteps=16, objective='pred_noise'))

        diff_util.set_dataloader(data_loader)
        diff_util.train(epochs=50, model_name="bedroom_16_64x64_pnoise")

    elif dst == 'Church':
        data_loader = load_dataset(batch_size=64, dataset='Church', dataset_dir=cfg.DATASET_DIR)
        visualize(load_dataset(batch_size=32, dataset='Church', dataset_dir=cfg.DATASET_DIR),
                  save_dir='./sampling_res/training_data_church.jpg')

        diff_util = DiffusionUtils(build_diffusion(image_size=64, timesteps=16, objective='pred_noise'))

        diff_util.set_dataloader(data_loader)
        diff_util.train(epochs=50, model_name="church_16_64x64_pnoise_scratch")

    elif dst == 'CelebA_HQ':
        data_loader = load_dataset(batch_size=64, dataset='CelebA_HQ', dataset_dir=cfg.DATASET_DIR)
        visualize(load_dataset(batch_size=32, dataset='CelebA_HQ', dataset_dir=cfg.DATASET_DIR),
                  save_dir='sampling_res/training_data_celeba_hq.jpg')

        diff_util = DiffusionUtils(build_diffusion(image_size=32, timesteps=16, objective='pred_v'))

        diff_util.set_dataloader(data_loader)
        # d_dir = 'saved_models/CelebA_HQ_128x128_1024/diffusion_celeba_hq_1024_128x128_epoch_10.pth'
        # diff_util.load_trained_model(d_dir)
        diff_util.train(epochs=50, start_epochs=0, model_name="celeba_hq_16_32x32_pv")

    elif dst == 'Cat':
        data_loader = load_dataset(batch_size=100, dataset='Cat', dataset_dir=cfg.DATASET_DIR)
        visualize(load_dataset(batch_size=32, dataset='Cat', dataset_dir=cfg.DATASET_DIR),
                  save_dir='./sampling_res/training_data_cat.jpg')

        diff_util = DiffusionUtils(build_diffusion(image_size=32, timesteps=16, objective='pred_v'))

        diff_util.set_dataloader(data_loader)
        diff_util.train(epochs=20, model_name="cat_16_32x32")
    elif dst == 'Cifar10':
        data_loader = load_dataset(batch_size=50, dataset='Cifar10', dataset_dir=cfg.DATASET_DIR)
        visualize(load_dataset(batch_size=32, dataset='Cifar10', dataset_dir=cfg.DATASET_DIR),
                  save_dir='./sampling_res/training_data_cifar10.jpg')

        diff_util = DiffusionUtils(build_diffusion(image_size=32, timesteps=4, objective='pred_v'))

        diff_util.set_dataloader(data_loader)
        # d_dir = 'saved_models/diffusion_CelebA_1024_64x64_epoch_5.pth'
        # diff_util.load_trained_model(d_dir)
        diff_util.train(epochs=10, start_epochs=0, model_name="Cifar10_32x32_4_pv_tscratch")
    elif dst == 'Imagenet':
        data_loader = load_dataset(batch_size=50, dataset='Imagenet', dataset_dir=cfg.DATASET_DIR)
        visualize(load_dataset(batch_size=32, dataset='Imagenet', dataset_dir=cfg.DATASET_DIR),
                  save_dir='./sampling_res/training_data_imagenet.jpg')

        diff_util = DiffusionUtils(build_diffusion(image_size=32, timesteps=64, objective='pred_v'))

        diff_util.set_dataloader(data_loader)
        d_dir = 'saved_models/diffusion_Imagenet_32x32_64_pv_tscratch_epoch_200_.pth'
        diff_util.load_trained_model(d_dir)
        diff_util.train(epochs=200, start_epochs=0, model_name="Imagenet_32x32_64_pv_tscratch")


def sampling():
    diff_util = DiffusionUtils(build_diffusion(image_size=128, timesteps=1024, sampling_timesteps=1024, objective="pred_v", using_ddim=True))
    d_dir = 'saved_models/Bedroom_128x128_1024_pv/diffusion_bedroom_1024_128x128_pv_epoch_50.pth'
    diff_util.load_trained_model(d_dir)
    diff_util.sample(res_id="bedroom_teacher", num_img=10, nrow=10)


def calc_fid():
    use_pdistill = True

    teacher = build_diffusion(image_size=64, timesteps=1024, sampling_timesteps=512, objective='pred_noise', using_ddim=False)

    # mapping_sequence = [0, 64, 128, 192, 496, 500, 504, 508, 512, 516, 520, 524, 528, 832, 896, 960]
    mapping_sequence = None
    student = build_diffusion(image_size=64, timesteps=8, objective='pred_noise', is_student=True,
                              teacher=teacher, use_pdistill=use_pdistill, mapping_sequence=mapping_sequence)

    diff_util = DiffusionUtils(teacher, student_diff=student)

    # d_dir = 'saved_models/CelebA_HQ_64x64_1024_pnoise/diffusion_celeba_hq_1024_64x64_pnoise_epoch_47.pth'
    # diff_util.load_trained_model(d_dir)

    s_dir = 'saved_models/student/celeba_pnoise_ours_8_64x64/diffusion_CelebA_HQ_64x64_8_student_pnoise_l1_sl_epoch_26.pth'
    # s_dir = 'saved_models/student/CelebA_HQ_64x64_pdistill/diffusion_CelebA_HQ_64x64_16_student_pnoise_pdistill_epoch_20.pth'
    # s_dir = 'saved_models/consistency_model/diffusion_ct_celeba_hq_64x64_16_ct_pnoise_epoch_50.pth'
    diff_util.load_trained_student(s_dir)

    data_loader = load_dataset(batch_size=100, dataset='CelebA_HQ', dataset_dir=cfg.DATASET_DIR)
    diff_util.set_dataloader(data_loader)

    diff_util.calc_fid(use_student=True)
    # diff_util.calc_fid(use_student=False)


def interpolate(x1, x2, lam):
    return (1 - lam) * x1 + lam * x2


def sampling_student():
    # sampling
    teacher = build_diffusion(image_size=128, timesteps=1024, objective='pred_v', using_ddim=True)

    student = build_diffusion(image_size=128, timesteps=128, objective='pred_v', sampling_timesteps=128,
                              is_student=True, teacher=teacher, using_ddim=True, use_pdistill=False)

    diff_util = DiffusionUtils(teacher, student_diff=student)

    d_dir = 'saved_models/CelebA_HQ_128x128_1024_pv/diffusion_celeba_hq_1024_128x128_pv_epoch_100.pth'
    diff_util.load_trained_model(d_dir)

    s_dir = 'saved_models/student/CelebA_HQ_128x128_128_pv_l1_sl/diffusion_CelebA_HQ_128x128_128_student_pv_l1_sl_epoch_32.pth'
    # s_dir = 'saved_models/student/church_pdistill_64x64_pnoise/diffusion_Church_64x64_128_student_pnoise_pdistill_epoch_8.pth'
    diff_util.load_trained_student(s_dir)

    num_img = 1
    shape = (num_img, student.channels, student.image_size, student.image_size)
    input_noise = torch.randn(shape, device=student.device)

    sampled_img = diff_util.sample(res_id="agif", num_img=num_img, nrow=10, use_student=True, input_noise=input_noise)
    sampled_img = diff_util.sample(res_id="bgif", num_img=num_img, nrow=10, use_student=False, input_noise=input_noise)


def train_student():
    dst = 'Bedroom'
    # dst = None
    if dst == 'Cifar10':
        teacher = build_diffusion(image_size=32, timesteps=1024, objective='pred_noise')
        student = build_diffusion(image_size=32, timesteps=1024, objective='pred_noise', is_student=True, teacher=teacher)

        diff_util = DiffusionUtils(teacher, student_diff=student)

        d_dir = 'saved_models/cifar10_1024/diffusion_Cifar10_32x32_1024_pnoise_epoch_50.pth'
        # d_dir = 'saved_models/diffusion_Cifar10_32x32_32_student_pnoise_pdistill_epoch_5.pth'
        diff_util.load_trained_model(d_dir)

        data_loader = load_dataset(batch_size=50, dataset='Cifar10', dataset_dir=cfg.DATASET_DIR)
        diff_util.set_dataloader(data_loader)

        # diff_util.train_student(epochs=50, start_epochs=0, model_name="Cifar10_32x32_128_student_pnoise_l1_sl")
        # diff_util.train_student(epochs=20, start_epochs=0, model_name="Cifar10_32x32_16_student_pnoise_cdistill", loss_type="cdistill")
        # diff_util.train_student(epochs=5, start_epochs=0, model_name="Cifar10_32x32_16_student_pnoise_pdistill", loss_type="pdistill")
        # diff_util.train_student(epochs=10, start_epochs=0, model_name="Cifar10_32x32_128_student_pnoise_scratch", loss_type="scratch")

    elif dst == 'CelebA_HQ':
        teacher = build_diffusion(image_size=64, timesteps=1024, objective='pred_noise')

        # mapping_sequence = [0, 64, 128, 192, 496, 500, 504, 508, 512, 516, 520, 524, 528, 832, 896, 960]
        # mapping_sequence = [0, 64, 128, 192, 256, 320, 504, 508, 512, 516, 520, 704, 768, 832, 896, 960]
        mapping_sequence = [0, 64, 128, 192, 256, 320, 384, 508, 512, 516, 640, 704, 768, 832, 896, 960]
        # mapping_sequence = None
        student = build_diffusion(image_size=64, timesteps=16, objective='pred_noise', is_student=True,
                                  teacher=teacher, use_pdistill=False, mapping_sequence=mapping_sequence)

        diff_util = DiffusionUtils(teacher, student_diff=student)

        d_dir = 'saved_models/CelebA_HQ_64x64_1024_pnoise/diffusion_celeba_hq_1024_64x64_pnoise_epoch_47.pth'
        # d_dir = 'saved_models/CelebA_HQ_128x128_1024_pv/diffusion_celeba_hq_1024_128x128_pv_epoch_100.pth'
        # d_dir = 'saved_models/diffusion_CelebA_HQ_64x64_32_student_pnoise_pdistill_epoch_20.pth'
        diff_util.load_trained_model(d_dir)

        data_loader = load_dataset(batch_size=40, dataset='CelebA_HQ', dataset_dir=cfg.DATASET_DIR)
        diff_util.set_dataloader(data_loader)

        s_dir = 'saved_models/student/CelebA_HQ_64x64_16_pnoise_l1_sl/diffusion_CelebA_HQ_64x64_16_student_pnoise_l1_sl_epoch_50.pth'
        diff_util.load_trained_student(s_dir)
        diff_util.train_student(epochs=5, start_epochs=0, model_name="CelebA_HQ_64x64_4_student_pnoise_l1_sl_40percent")
        # diff_util.train_student(epochs=20, start_epochs=0, model_name="CelebA_HQ_64x64_4_student_pnoise_pdistill", loss_type="pdistill")
    elif dst == 'Church':
        teacher = build_diffusion(image_size=128, timesteps=1024, objective='pred_v')

        mapping_sequence = [int(i*(10.24)) for i in range(100)]
        student = build_diffusion(image_size=128, timesteps=100, objective='pred_v', is_student=True, teacher=teacher,
                                  use_pdistill=False, mapping_sequence=mapping_sequence)

        diff_util = DiffusionUtils(teacher, student_diff=student)

        # d_dir = 'saved_models/church_64x64_1024_pnoise/diffusion_church_1024_64x64_pnoise_epoch_50.pth'
        # d_dir = 'saved_models/diffusion_Church_64x64_32_student_pnoise_pdistill_epoch_9.pth'
        d_dir = 'saved_models/church_1024_128x128_pv/diffusion_church_1024_128x128_pv_epoch_50.pth'
        diff_util.load_trained_model(d_dir)

        data_loader = load_dataset(batch_size=40, dataset='Church', dataset_dir=cfg.DATASET_DIR)
        diff_util.set_dataloader(data_loader)

        diff_util.train_student(epochs=50, start_epochs=0, model_name="Church_128x128_100_student_pv_l1_sl")
        # diff_util.train_student(epochs=50, start_epochs=0, model_name="Church_64x64_16_student_pnoise_pdistill", loss_type="pdistill")
    elif dst == 'CelebA':
        teacher = build_diffusion(image_size=64, timesteps=1024, objective='pred_noise')
        student = build_diffusion(image_size=64, timesteps=16, objective='pred_noise', is_student=True, teacher=teacher)

        diff_util = DiffusionUtils(teacher, student_diff=student)

        d_dir = 'saved_models/celeba_pnoise_64x64_1024/diffusion_CelebA_1024_64x64_epoch_30.pth'
        diff_util.load_trained_model(d_dir)

        data_loader = load_dataset(batch_size=25, dataset='CelebA', dataset_dir=cfg.DATASET_DIR)
        diff_util.set_dataloader(data_loader)

        s_dir = 'saved_models/diffusion_CelebA_64x64_16_student_pnoise_l1_sl_epoch_10.pth'
        diff_util.load_trained_student(s_dir)
        diff_util.train_student(epochs=20, start_epochs=10, model_name="CelebA_64x64_16_student_pnoise_l1_sl")
    elif dst == 'Bedroom':
        teacher = build_diffusion(image_size=128, timesteps=1024, objective='pred_v')

        # mapping_sequence = [int(i * (10.24)) for i in range(100)]
        mapping_sequence = None
        student = build_diffusion(image_size=128, timesteps=2, objective='pred_v', is_student=True, teacher=teacher, use_pdistill=False, mapping_sequence=mapping_sequence)

        # student = build_diffusion(image_size=128, timesteps=1024, objective='pred_v', is_student=True, teacher=teacher,
        #                           use_pdistill=True, mapping_sequence=None)

        diff_util = DiffusionUtils(teacher, student_diff=student)

        # d_dir = 'saved_models/church_64x64_1024_pnoise/diffusion_church_1024_64x64_pnoise_epoch_50.pth'
        # d_dir = 'saved_models/diffusion_Church_64x64_32_student_pnoise_pdistill_epoch_9.pth'
        d_dir = 'saved_models/Bedroom_128x128_1024_pv/diffusion_bedroom_1024_128x128_pv_epoch_50.pth'
        diff_util.load_trained_model(d_dir)

        data_loader = load_dataset(batch_size=40, dataset='Bedroom', dataset_dir=cfg.DATASET_DIR)
        diff_util.set_dataloader(data_loader)

        diff_util.train_student(epochs=50, start_epochs=0, model_name="Bedroom_128x128_2_student_pv_l1_sl")


if __name__ == '__main__':
    # train_diffusion()

    # train_ct_diffusion()

    # train_student()
    # sampling_student()
    sampling()

    # calc_fid()

# def sampling_student():
#     # sampling
#     teacher = build_diffusion(image_size=128, timesteps=1024, objective='pred_v', using_ddim=True)
#
#     # mapping_sequence = [int(i * (10.24)) for i in range(100)]
#     mapping_sequence = None
#     student = build_diffusion(image_size=128, timesteps=128, objective='pred_v', sampling_timesteps=100,
#                               is_student=True, teacher=teacher, using_ddim=True, use_pdistill=False, mapping_sequence=mapping_sequence)
#
#     diff_util = DiffusionUtils(teacher, student_diff=student)
#
#     d_dir = 'saved_models/CelebA_HQ_128x128_1024_pv/diffusion_celeba_hq_1024_128x128_pv_epoch_100.pth'
#     diff_util.load_trained_model(d_dir)
#
#     s_dir = 'saved_models/student/CelebA_HQ_128x128_128_pv_l1_sl/diffusion_CelebA_HQ_128x128_128_student_pv_l1_sl_epoch_33.pth'
#     # s_dir = 'saved_models/student/church_pdistill_64x64_pnoise/diffusion_Church_64x64_128_student_pnoise_pdistill_epoch_8.pth'
#     diff_util.load_trained_student(s_dir)
#
#     num_img = 9
#     shape = (num_img, student.channels, student.image_size, student.image_size)
#     # input_noise = torch.randn(shape, device=student.device)
#
#     input_noise = torch.load('./sampling_res/student/CelebA_HQ_128x128_128_pv_l1_sl/noise_celeba_hq_128x128.pth')
#
#     x1 = input_noise[0]
#     x2 = input_noise[8]
#     x3 = interpolate(x1, x2, lam=1.0 / 8)
#     x4 = interpolate(x1, x2, lam=2.0 / 8)
#     x5 = interpolate(x1, x2, lam=3.0 / 8)
#     x6 = interpolate(x1, x2, lam=4.0 / 8)
#     x7 = interpolate(x1, x2, lam=5.0 / 8)
#     x8 = interpolate(x1, x2, lam=6.0 / 8)
#     x9 = interpolate(x1, x2, lam=7.0 / 8)
#     input_noise = torch.stack([x1, x3, x4, x5, x6, x7, x8, x9, x2], dim=0)
#
#     # i = 20
#     # torch.save(input_noise, './sampling_res/input_noise{}.pth'.format(i))
#
#     # torch.manual_seed(1105)
#     # torch.cuda.manual_seed(1105)
#
#     sampled_img = diff_util.sample(res_id="CelebA_HQ_interpolation_student2", num_img=num_img, nrow=9, use_student=True, input_noise=input_noise)
#     sampled_img = diff_util.sample(res_id="CelebA_HQ_interpolation_teacher2", num_img=num_img, nrow=9, use_student=False, input_noise=input_noise)
