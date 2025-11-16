import torch
import torchvision


class DiffusionUtils(object):

    def __init__(self, diffusion, student_diff=None, data_loader=None, autoencoder=None):
        self.diffusion = diffusion
        self.data_loader = data_loader
        self.trained_model_dir = None

        self.trained_student_dir = None

        self.student_diff = student_diff

        self.autoencoder = autoencoder

    def set_dataloader(self, data_loader, mean_std=None):
        self.data_loader = data_loader
        self.mean_std = mean_std

    def load_trained_model(self, trained_model_dir):
        self.trained_model_dir = trained_model_dir
        self.diffusion.load_state_dict(torch.load(self.trained_model_dir))

    def load_trained_student(self, trained_student_dir):
        self.trained_student_dir = trained_student_dir
        self.student_diff.load_state_dict(torch.load(self.trained_student_dir))

        # self.student_diff.model = self.diffusion.model

    def train(self, epochs=1, start_epochs=0, model_name=""):

        optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=0.0002)

        for e in range(start_epochs, epochs):
            print('epoch: ', e + 1, ' / ', epochs)
            i = 0
            for training_images, y in self.data_loader:

                optimizer.zero_grad()

                loss = self.diffusion(training_images.cuda())

                loss.backward()

                optimizer.step()

                i += 1
                if i % 100 == 0:
                    print('iter: ', i, " / ", len(self.data_loader), " loss: ", loss.item())

            if (e+1) % 10 == 0:
                torch.save(self.diffusion.state_dict(), './saved_models/diffusion_{}_epoch_{}.pth'.format(model_name, e+1))
            self.sample(res_id=e + 1, num_img=64, nrow=8)

    def sample(self, res_id=0, num_img=8, nrow=4, save_dir='./sampling_res/gif/res_{}.jpg', use_student=False, input_noise=None, is_consistency_model=False):

        if not use_student:
            if not is_consistency_model:
                sampled_images = self.diffusion.sample(batch_size=num_img, input_noise=input_noise)
            else:
                sampled_images = self.diffusion.sample_with_consistency_sampling(batch_size=num_img, input_noise=input_noise)
        else:
            if not is_consistency_model:
                sampled_images = self.student_diff.sample(batch_size=num_img, input_noise=input_noise)
            else:
                sampled_images = self.student_diff.sample_with_consistency_sampling(batch_size=num_img, input_noise=input_noise)
        print("sampled_images shape", sampled_images.shape)
        torchvision.utils.save_image(sampled_images, "./sampling_res/res_{}.jpg".format(res_id), nrow=nrow, padding=2)
        torch.save(sampled_images, "./sampling_res/res_{}.pth".format(res_id))

        return sampled_images

    def train_student(self, epochs=1, start_epochs=0, model_name="", loss_type=None):

        optimizer = torch.optim.Adam(self.student_diff.parameters(), lr=0.0002)

        self.sample(res_id=0, num_img=32, nrow=8, save_dir='./sampling_res/res_{}_s.jpg', use_student=True)
        for e in range(start_epochs, epochs):
            print('epoch: ', e + 1, ' / ', epochs)
            i = 0
            for training_images, y in self.data_loader:

                optimizer.zero_grad()

                if loss_type is None:
                    loss = self.student_diff.student_loss(training_images.cuda(), self.diffusion)
                else:
                    loss = self.student_diff(training_images.cuda())

                loss.backward()

                optimizer.step()

                i += 1
                if i % 100 == 0:
                    print('iter: ', i, " / ", len(self.data_loader), " loss: ", loss.item())

            torch.save(self.student_diff.state_dict(), './saved_models/diffusion_{}_epoch_{}.pth'.format(model_name, e+1))

            # self.sample(res_id=e + 1)
            self.sample(res_id=e + 1, num_img=32, nrow=8, save_dir='./sampling_res/res_{}_s.jpg', use_student=True)



