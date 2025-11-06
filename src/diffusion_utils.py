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

                # if i % 1000 == 0:
                #     self.sample(res_id="iters_{}".format(i))

            if (e+1) % 10 == 0:
                torch.save(self.diffusion.state_dict(), './saved_models/diffusion_{}_epoch_{}.pth'.format(model_name, e+1))
            self.sample(res_id=e + 1, num_img=64, nrow=8)

    def train_ct(self, epochs=1, start_epochs=0, model_name=""):

        optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=0.0002)

        for e in range(start_epochs, epochs):
            print('epoch: ', e + 1, ' / ', epochs)
            i = 0
            for training_images, y in self.data_loader:

                optimizer.zero_grad()

                loss = self.diffusion.ct_loss(training_images.cuda(), e, i)

                loss.backward()

                optimizer.step()

                i += 1
                if i % 100 == 0:
                    print('iter: ', i, " / ", len(self.data_loader), " loss: ", loss.item())

            if (e+1) % 1 == 0:
                torch.save(self.diffusion.state_dict(), './saved_models/diffusion_ct_{}_epoch_{}.pth'.format(model_name, e+1))
            self.sample(res_id=e + 1, num_img=64, nrow=8, save_dir="./sampling_res_new/res_ct_{}.jpg")

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
        torchvision.utils.save_image(sampled_images, "./sampling_res/res_{}.jpg".format(res_id), nrow=10, padding=2)
        torch.save(sampled_images, "./sampling_res/res_{}.pth".format(res_id))
        # for i in range(sampled_images.shape[1]):
        #     res_name = res_id + "_{}".format(i)
        #     torchvision.utils.save_image(sampled_images[0][i], save_dir.format(res_name), nrow=nrow, padding=2)
        # torchvision.utils.save_image(sampled_images[0][sampled_images.shape[1]-1], save_dir.format("aaa{}".format(sampled_images.shape[1])), nrow=nrow, padding=2)
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
                elif loss_type == "pdistill":
                    loss = self.student_diff.pdistill_student_loss(training_images.cuda(), self.diffusion)
                elif loss_type == "cdistill":
                    loss = self.student_diff.cdistillation_student_loss(training_images.cuda(), self.diffusion)
                else:  # loss_type == "scratch"
                    loss = self.student_diff(training_images.cuda())

                loss.backward()

                optimizer.step()

                i += 1
                if i % 100 == 0:
                    print('iter: ', i, " / ", len(self.data_loader), " loss: ", loss.item())

                # if i % 1000 == 0:
                #     self.sample(res_id="iters_{}".format(i))

            torch.save(self.student_diff.state_dict(), './saved_models/diffusion_{}_epoch_{}.pth'.format(model_name, e+1))

            # self.sample(res_id=e + 1)
            self.sample(res_id=e + 1, num_img=32, nrow=8, save_dir='./sampling_res/res_{}_s.jpg', use_student=True)

    def train_autoencoder(self, epochs=10, start_epochs=0, model_name=""):
        optimizer = torch.optim.Adam(self.student_diff.parameters(), lr=0.0002)

        import torch.nn.functional as F

        for e in range(start_epochs, epochs):
            print('epoch: ', e + 1, ' / ', epochs)
            i = 0
            for training_images, y in self.data_loader:
                optimizer.zero_grad()
                z = self.autoencoder.encode(training_images)
                recover_images = self.autoencoder.decode(z)

                loss = F.l1_loss(recover_images, training_images, reduction="mean")

                loss.backward()

                optimizer.step()

                i += 1
                if i % 100 == 0:
                    print('iter: ', i, " / ", len(self.data_loader), " loss: ", loss.item())

            torch.save(self.student_diff.state_dict(),'./saved_models/autoencoder_{}_epoch_{}.pth'.format(model_name, e + 1))

    def calc_fid(self, use_student=True):
        from torchmetrics.image.fid import FrechetInceptionDistance

        torch.manual_seed(123)
        torch.cuda.manual_seed(123)

        # n_feature = 192  # 64, 192, 768, 2048

        num_64 = 0
        num_192 = 5
        num_768 = 5

        fid_list = []
        i = 0
        for x, y in self.data_loader:

            if i < num_64:
                fid = FrechetInceptionDistance(feature=64).cuda()
            elif i < (num_64+num_192):
                fid = FrechetInceptionDistance(feature=192).cuda()
            else:
                fid = FrechetInceptionDistance(feature=768).cuda()

            torchvision.utils.save_image(x, './sampling_res_new/fid_real.jpg', nrow=8, padding=2)

            x = (x * 255).type(torch.uint8)
            x = x.cuda()

            if not use_student:
                syn_imgs_o = self.diffusion.sample(batch_size=x.shape[0])
            else:
                syn_imgs_o = self.student_diff.sample(batch_size=x.shape[0])

            torchvision.utils.save_image(syn_imgs_o, './sampling_res_new/fid_vidualization.jpg', nrow=8, padding=2)

            syn_imgs = (syn_imgs_o * 255).type(torch.uint8)
            syn_imgs = syn_imgs.cuda()

            fid.update(x, real=True)
            fid.update(syn_imgs, real=False)
            res = fid.compute()
            fid_list.append(res.item())

            i += 1
            if i == num_192+num_64+num_768:
                torchvision.utils.save_image(syn_imgs_o, './sampling_res/fid_test.jpg', nrow=8, padding=2)
                break
            print(fid_list)
        print(fid_list, len(fid_list))
        print(sum(fid_list)/len(fid_list))



