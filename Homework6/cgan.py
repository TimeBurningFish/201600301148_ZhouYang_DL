#coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset,TensorDataset,DataLoader
from torchvision import transforms, utils
from torchvision.utils import save_image,make_grid
from torch.autograd import Variable

import numpy as np

import os
# import paras
import pandas as pd
from matplotlib import pyplot as plt



# 超参数
gpu_id = '0'
if gpu_id is not None:
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
	device = torch.device('cuda')
else:
	device = torch.device('cpu')
if os.path.exists('cgan_images') is False:
	os.makedirs('cgan_images')

FloatTensor = torch.cuda.FloatTensor if gpu_id else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if gpu_id else torch.LongTensor

# z_dim = paras.z_dim
# batch_size = paras.batch_size
# learning_rate = paras.learning_rate
# total_epochs = paras.total_epochs

z_dim = 100
batch_size = 100
learning_rate = 3e-4
total_epochs = 200

n_classes = 10
img_shape = (1,28,28)


class Discriminator(nn.Module):
	'''全连接判别器，用于1x28x28的MNIST数据,输出是数据和类别'''
	def __init__(self):
		super(Discriminator, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(794, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),
			nn.Linear(1024, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),
			nn.Linear(512, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(0.3),
			nn.Linear(256, 1),
			nn.Sigmoid()
		)

		#your code

	def forward(self, x, c):#c = class
		x = x.view(x.size(0), 784)
		x = torch.cat([x, c], 1)
		out = self.model(x)
		return out
		#your code

class Generator(nn.Module):	
	'''全连接生成器，用于1x28x28的MNIST数据，输入是噪声和类别'''
	def __init__(self, z_dim):
		super(Generator, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(10 + z_dim, 256),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 512),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 784),
			nn.Tanh()
		)
		#your code

	def forward(self, z, c):
		## z 是 noise
		z = z.view(z.size(0), z_dim)
		x = torch.cat([z, c], 1)
		out = self.model(x)
		return out.view(x.size(0), 28, 28)
		#your code


def one_hot(labels, class_num):	
	'''把标签转换成one-hot类型'''
	

	labels = labels.cpu()
	ones = torch.zeros(batch_size, class_num)
	
	ones = ones.scatter_(1, labels, 1).to(device)
	return ones



# 初始化构建判别器和生成器
discriminator = Discriminator().to(device)
generator = Generator(z_dim=z_dim).to(device)

# 初始化二值交叉熵损失
bce = torch.nn.BCELoss().to(device)
ones = torch.ones(batch_size).to(device)
zeros = torch.zeros(batch_size).to(device)

# 初始化优化器，使用Adam优化器
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=[0.5, 0.999])
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=[0.5, 0.999])

# 加载fashion数据集

mean = 0.5
std = 0.5
normalize = transforms.Normalize(
	mean=[72.9568306122449],
	std=[89.96686298512124]
)


preprocess = transforms.Compose([
        transforms.ToTensor(),
		transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Normalize(mean=(mean,mean,mean), std=(std,std,std))
])


# preprocess = transforms.Compose([
#     #transforms.Scale(256),
#     #transforms.CenterCrop(224),
# 	transforms.ToTensor(),
# 	transforms.Normalize(mean=(0.5), std=(0.5))
# ])

class FashionMnists(Dataset):
	def __len__(self) -> int:
		return len(self.x)
	def __init__(self, csv_file: str,  transform=None) -> None:
		super().__init__()
		self.landmarks_frame = pd.read_csv(csv_file).values
		self.x =  self.landmarks_frame[:,1:].astype('uint8').reshape(-1,28,28)
		self.y =  self.landmarks_frame[:,:1]
		self.transform = transform

	def __getitem__(self, index:int):
		image = self.x[index]
		if self.transform:
			image = self.transform(image)
		image= image[0].reshape(-1,)
		label = self.y[index]
		return image, label


train_dataset = FashionMnists('fashion-mnist_train.csv', preprocess)
test_dataset = FashionMnists('fashion-mnist_test.csv', preprocess)

train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size,shuffle=True,num_workers=4)
test_loader = DataLoader(dataset = test_dataset, batch_size=batch_size,shuffle=True,num_workers=4)

#############################################################
#your code
#############################################################

#用于生成效果图
# 生成100个向量
fixed_c = torch.FloatTensor(100, 10).zero_()
fixed_c = fixed_c.scatter_(dim=1, index=torch.LongTensor(np.array(np.arange(0, 10).tolist()*10).reshape([100, 1])), value=1)
fixed_c = fixed_c.to(device)
# 生成100个随机噪声向量
fixed_z = torch.randn([100, z_dim]).to(device)

print("eval",fixed_z.shape,fixed_c.shape)

gls= []
dls = []
epochs = []



# 开始训练，一共训练total_epochs
for epoch in range(total_epochs):

	# 在训练阶段，把生成器设置为训练模式；对应于后面的，在测试阶段，把生成器设置为测试模式
	generator = generator.train()

	# 训练一个epoch
	for i, data in enumerate(train_loader):
		

		# 加载真实数据
		###############################
		#your code
		
		(imgs,labels) = data	

		real_imgs = Variable(imgs.type(FloatTensor))
		labels = Variable(labels.type(LongTensor))	

		###############################

		# 把对应的标签转化成 one-hot 类型
		################################
		#your code
		
		##这里使用embedding不使用onehot

		################################

		# 生成数据
		# 用正态分布中采样batch_size个随机噪声

		valid = torch.ones(batch_size).to(device)
		fake = torch.zeros(batch_size).to(device)


		######################训练D
		d_optimizer.zero_grad()
		
		labels = one_hot(labels,n_classes)
        # real 图像 loss
		validity_real = discriminator(real_imgs, labels).squeeze()
		# print(validity_real.shape,valid.shape)
		d_real_loss = bce(validity_real, valid)
 

        # fake 图像 loss
		z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, z_dim))))	
				# 生成 batch_size 个标签
		gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))
		gen_labels = gen_labels.unsqueeze(1)
		gen_labels = one_hot(gen_labels,n_classes)## 转成 one-hot

		gen_imgs = generator(z, gen_labels)		
		validity_fake = discriminator(gen_imgs, gen_labels).squeeze()
		d_fake_loss = bce(validity_fake, fake)

		d_loss = (d_real_loss + d_fake_loss) / 2


		d_loss.backward()
		d_optimizer.step()

		##################训练G
		# for _ in range(10):
		g_optimizer.zero_grad()

		# 生成数据		
		z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, z_dim))))			
		gen_imgs = generator(z, gen_labels)

		validity = discriminator(gen_imgs, gen_labels).squeeze()
		# 计算判别器损失，并优化判别器
		
		g_loss = bce(validity, valid)

		g_loss.backward()
		g_optimizer.step()



		if i%200 == 0:
			print(
				"[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
				% (epoch, total_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
			)


		# 输出损失 参考下方 print
		#print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, total_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))
	# 把生成器设置为测试模型，生成效果图并保存
	dls.append(float(d_loss))
	gls.append(float(g_loss))
	epochs.append(epoch)

	generator = generator.eval()
	fixed_fake_images = generator(fixed_z, fixed_c)
	# save_image(fixed_fake_images.reshape(fixed_fake_images.shape[0],28,28), 'cgan_images/{}.png'.format(epoch), nrow=10, normalize=True)

	plt.figure()
	raise ValueError(fixed_fake_images.shape)	
	grid = make_grid(fixed_fake_images.unsqueeze(1).data, nrow=10, normalize=True).permute(1, 2, 0).cpu().numpy()
	plt.imshow(grid)
	plt.axis('off')	
	plt.savefig('cgan_images/{}.png'.format(epoch))

	plt.figure()
	plt.plot(epochs, dls, label = "d")
	plt.plot(epochs, gls, label = "g")
	plt.legend()
	plt.savefig("loss.png")
	