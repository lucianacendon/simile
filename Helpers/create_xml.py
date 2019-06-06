##
#
# This code helps create the XML files required by the SIMILE library. 
# The code simply lists all pickle files inside a specified directory and write their local paths to a .XML file
#
# You should call this script using the following format: :
# 	> python create_xml.py -file_dir 'Path/to/dir' -out_dir '/Path/to/output_dir/' -filename filename.xml
#
##

import os
import argparse 


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-file_dir', help = 'Directory containing files to be listed on the XML file')
	parser.add_argument('-out_dir', help = 'Directory where you wish to save the output XML file')
	parser.add_argument('-filename', help = 'Name of the output file (Note: include ".xml" extension')

	try:
		args = parser.parse_args()
	except:
		print ('You should call this script using the following format: ')
		print ("	> python create_xml.py -file_dir 'Path/to/files_dir/' -out_dir '/Path/to/xml_output_dir/' -filename filename.xml")
		exit()

	if not (args.file_dir) or not (args.out_dir) or not (args.filename):
		print ('You should call this script using the following format: ')
		print ("	> python create_xml.py -file_dir Path/to/files_dir/ -out_dir /Path/to/xml_output_dir/ -filename filename.xml")
		exit()

	file_paths = [os.path.abspath(str(args.file_dir)+file) for file in os.listdir(str(args.file_dir)) if file.split('.')[1] == 'p']

	if (args.out_dir[-1] != '/'):
		args.out_dir = args.out_dir + '/'

	if (args.file_dir[-1] != '/'):
		args.file_dir = args.file_dir + '/'

	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	out_file = open(args.out_dir + str(args.filename), 'w')

	for path in file_paths:
		out_file.writelines(path+ '\n')

	out_file.close()