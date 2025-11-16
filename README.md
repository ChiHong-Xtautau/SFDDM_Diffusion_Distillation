# SFDDM

This is the code for the algorithm proposed by our paper:

"Chi Hong, Jiyue Huang, Robert Birke, Dick Epema, Stefanie Roos, and Lydia Y. Chen.
"Single-fold Distillation for Diffusion models." In European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD), 2025."

This project relies on https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main to implement diffusion models. To facilitating users, we provide a copy in this repo.

Then you will get the pretrained diffusion model on imagenet, and you can run the experiments. You may replace the pretrained models by yours.

An example of running the algorithm is shown in "run.py".

# Before running
To run the algorithm, please extract the pretrained diffusion model in "./trained_model". Please use the command
- sudo apt install p7zip-full
- cd trained_model
- 7z x model.7z.001

# To run this file
The project is developed under python 3.8.10

- pip install -r requirements.txt
- python run.py

# Expected Results
After running the example "run.py", we can get the following expected Results. Please note that due to randomness, the final results you have may differ slightly from what is shown here.

- the distilled student generators will be saved in "./saved_models"
- the sampling results from the student will be saved in "./sampling_res"