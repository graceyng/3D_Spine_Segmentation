import os
import re



##### So prob grab input from 

def get_inputs()






~~~~~~



file_name = 'params_DL.txt'

# Reading input file
inputDirPath = os.getcwd()
print(inputDirPath)

if not os.path.isfile(inputDirPath+"/"+file_name):
    print("Input file does not exist")
    print("New Input file created")
    input_file = open(file_name,"w")
    input_file.write("## Input parameters for Deep Learning Segmentation\n")
    input_file.write("## File reads between ':' and '#'\n")
    input_file.write("input_shape =      #(*insert*,*insert*)## size of input image matrix\n")
    input_file.write("output_shape =     #(*insert*,*insert*)## size of output image matrix\n")
    input_file.write("batch_size =      #*insert*## size of the batch. For 2D segmentation, seems to fail >28\n")
    input_file.write("epochs =     #*insert*## self explanatory\n")
    input_file.write("validation_steps =    #*insert*## no idea...?")

else:
	print("File imported . . . ")
	input_file = open(file_name,"r")
	lines = input_file.read().split("\n")



	counter = 0
	var = []
	data = []
	output = []

	for line in lines:
		if counter < 2 or counter >= 7:
			counter = counter + 1
			continue
		else:
			var_name = line.split(" =")[0]
			var.append(var_name)

			data_val = line.split(" #")

			if len( re.findall( r'\d+' , data_val[0] ) )==0:
				data.append(data_val[1])


				print('LKength was 0 at' + str(counter))
				print(data_val[1] )
			else:
				data.append(data_val[0])

			counter = counter + 1

print('data len is '+str(len(data)))

input_shape = int( re.findall(r'\d+', data[0]) [0]  )

output_shape = int( re.findall(r'\d+', data[1]) [0]  )

batch_size = int( re.findall(r'\d+', data[2]) [0]  )

epochs = int( re.findall(r'\d+', data[3]) [0]   )

validation_steps = int( re.findall(r'\d+', data[4]) [0]   )

print(' Input shape is ' + str(input_shape) + ' of type '  + str (type(input_shape)) )

print( ' Output shape is ' + str(input_shape) + ' of type '  + str (type(output_shape)) )

print( ' Batch size is ' + str(input_shape) + ' of type '  + str (type(batch_size)) )

print( ' Epoch size is ' + str(epochs) + ' of type '  + str (type(epochs)) )

print( ' Number of validation steps is ' + str(validation_steps) + ' of type '  + str (type(validation_steps)) )

print('CHECK')
print('')
print('')

tmp = re.findall( r'\d+', 'fkjd;lsaljf;das')
if len(tmp)==0:
	print('Length is 0')




