import sys
sys.path.append('Lib/')

import configparser
import argparse

from simile_train import *


def train_simile(configfile):

	simile = SIMILE_TRAIN (configfile)
	simile.train()

	print ('Training complete...!')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-config', default='config_train.ini', help='Config file with parameters for training')
	args = parser.parse_args()
	
	if not os.path.isfile(args.config):
		print ('Error: configfile', args.config, 'does not exist.')
		exit()

	train_simile(args.config)