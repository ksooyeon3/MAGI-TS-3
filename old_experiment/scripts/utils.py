import os
import numpy as np
from collections import OrderedDict

class seed(object):
	def __init__(self):
		self.seed = None

	def random(self, no_seed, random_seed):
		np.random.seed(random_seed)
		self.seed = np.random.randint(int(1e9), size=no_seed)
		# remove repeated seed
		while(self.seed.size < no_seed):
			new_seed = np.random.randint(int(1e9),size=no_seed-self.seed.size)
			self.seed = np.concatenate((self.seed,new_seed),0)
			self.seed = np.unique(self.seed)

	def load(self, input_path):
		self.seed = np.loadtxt(input_path).squeeze().astype(int)

	def get(self, idx):
		if (idx >= self.seed.size):
			raise ValueError("%d is out of range" %(idx))
		return (self.seed[idx])

	def replace(self, idx, value):
		if (idx >= self.seed.size):
			raise ValueError("%d is out of range" %(idx))
		self.seed[idx] = value

	def save(self, output_path):
		np.savetxt(output_path, self.seed, fmt='%s')

class params(object):
	def __init__(self):
		pass

	def read(self, input_path):
		self.params = OrderedDict()
		info = open(input_path, 'r').readlines()
		for i in range(len(info)):
			infoline = info[i].strip().replace(" ","")
			if (infoline != ''):
				key, value = infoline.split('=')
				group, subgroup = key.split(':')
				self.add(group, subgroup, value)

	def add(self, group, subgroup, value):
		if (group not in self.params):
			self.params[group] = OrderedDict()
		self.params[group][subgroup] = value

	def get(self, group, subgroup):
		if (group not in self.params):
			return (None)
		if (subgroup not in self.params[group]):
			return (None)
		return (self.params[group][subgroup])

	def save(self, output_dir):
		file = open(os.path.join(output_dir,'params.config'), 'w')
		for group in self.params:
			for subgroup in self.params[group]:
				value = self.params[group][subgroup]
				file.write('%s:%s=%s\n' %(group,subgroup,value))
			file.write('\n')
		file.close()

