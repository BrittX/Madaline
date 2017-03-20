"""
File for the madaline neural network
"""
import madmenu as mm
import random as r
import os
import time

"""
class for Training Net
"""
class Training(object):
	def __init__(self, inputs, outputs, pairs, weights, tsets):
		self.inputs = inputs
		self.outputs = outputs
		self.pairs = pairs
		tsets = [list(map(float, lst)) for lst in tsets] # Cast all the values in tsets to an int
		self.tset = tsets # Store training set and their outputs aka my s and t

		# Set initial weight to random value
		if weights == 1: 
			self.bias = [round(r.uniform(-0.5, 0.5), 2) for x in range(3)] # 3 bias' with user set weight
			# Create matrix to store w[i][j] values of weights
			self.weights = [[round(r.uniform(-0.5, 0.5), 2) for x in range(self.inputs)] for y in range(self.inputs)]
		# Set initial weight to 0
		else: 
			self.weights = [[weights for x in range(self.inputs)] for y in range(self.inputs)]# Store weight values as 0
			self.bias = [round(r.uniform(-0.5, 0.5), 2) for x in range(3)]

"""
Run the training algorithm for Madaline

args:
	tsets: training sets and their associated outputs (2D array)
	bias: Array of the initilized bias'
	rate: user determined learning rate of Madaline
	epochs: max # of times to run program
	weights: 2D array of initial weights for each training set
	pairs: Number of training pairs in the set
	inputs: Number of inputs in each training set
	outfile: output file to save the weights to
"""
def trainAlgo(tsets, bias, rate, epochs, weights, pairs, inputs, outfile, outputs):
	# Get random value for v1, v2 bias
	v1 = v2 = round(r.uniform(-0.5, 0.5), 2)
	print('Epochs equals: ', epochs)
	x = [] # to store x inputs
	t = [] # to store t outputs
	stop = False # for determining convergence
	a, b, c = 0, 1, 2 # for the bias/weights
	count = 0 # keep track of number of inputs we've done so far
	era = 0 # Keep track of number of epochs
	converged = 0
	outs = []

	# Store each training pair, s:t 
	for i,tset in enumerate(tsets):
		if i % 3 == 1: # to get the inputs
			x.append(tset)
		if i % 3 == 2: # to get the outputs
			t.append(tset)
	while not stop:
		# Get the input to the hidden layer
		for i in range(pairs):
			print('\nThese are the weights: ', weights)
			print('These are the bias: ', bias)
			count += 1
			z_in1 = round(bias[a] + (x[i][a] * weights[a][a]) + (x[i][b] * weights[b][a]), 2) # Need to move to for loop(?)
			z_in2 = round(bias[b] + (x[i][a] * weights[a][b]) + (x[i][b] * weights[b][b]), 2)
			print('This is z_in1: ', z_in1)
			print('This is z_in2: ', z_in2)
			# Find output of hidden layers
			z1 = activateF(z_in1)
			z2 = activateF(z_in2)
			
			# Get output of this hidden layer
			y_in = round(bias[c] + (z1 * v1) + (z2 * v2), 2)
			
			# Get y = f(y_in)
			y = float(activateF(y_in))
			print('This is y', y)
			print('The t to check is: ', t[i][a])
			
			# Check if error occured
			if t[i][a] == y: # Gets corresponding output
				print('t == y') 
				change = False
				converged +=1 # increment converged
				print('Converged equals ', converged)
				# Check if we've converged
				if converged >= pairs: # Means we haven't updated weights for each pair
					print('Training converged after {x} epochs'.format(x=era))
					stop = True
					# Write to output file
					with open(outfile, "w") as store:
						store.write("%d\n%d\n%d\n\n%s\n%s\n%.2f\n%.2f" %(inputs, outputs, pairs, weights, bias, v1, v2))
					store.close()
					return outfile
					break
				continue
			# t = 1 and z1/z2 = -1
			elif t[i][a] == 1.0 and (z1 == -1 and z2 == -1): # check if t = 1 and z1/z2 = -1
				val = min((z_in1, z_in2), key=lambda x: abs(x - 0)) # Get z value closest to 0
				print('This is the minimum value: ', val)
				if val == z_in1:
					# Update bias and weight values
					bias[a] = round(bias[a] + rate * (1 - z_in1), 2)
					weights[a][a] = round(weights[a][a] + rate * (1 - z_in1) * x[i][a], 2)
					weights[b][a] = round(weights[b][a] + rate * (1 - z_in1) * x[i][b], 2)
					print('T = 1, Updated bias[0]: ', bias[a])
					print('T = 1, Updated weights: ', weights)
				if val == z_in2:
					# Update the weight and bias corresponding to z2
					bias[b] = round(bias[b] + rate * (1 - z_in2), 2)
					weights[b][b] = round(weights[b][b] + rate * (1 - z_in2) * x[i][b], 2)
					weights[a][b] = round(weights[a][b] + rate * (1 - z_in2) * x[i][a], 2)
					print('T = 1, Updated bias[1]: ', bias[b])
					print('T = 1, Updated weights: ', weights)
				# Reset converged
				if converged > 0: 
					converged = 0
			# t = -1 and z1 0r z2 = 1
			elif t[i][a] == -1.0 and (z1 == 1 or z2 ==1):
				# Both z_in1 and z_in2 have positive inputs
				if z_in1 >= 0 and z_in2 >= 0:
					print('both positive')
					# update bias'
					bias[a] = round(bias[a] + rate * (-1 - z_in1), 2)
					bias[b] = round(bias[b] + rate * (-1 - z_in2), 2)
					# update weights
					weights[a][a] = round(weights[a][a] + rate * (-1 - z_in1) * x[i][a], 2)
					weights[a][b] = round(weights[a][b] + rate * (-1 - z_in2) * x[i][a], 2)
					weights[b][b] = round(weights[b][b] + rate * (-1 - z_in2) * x[i][b], 2)
					weights[b][a] = round(weights[b][a] + rate * (-1 - z_in1) * x[i][b], 2)
				elif z_in1 >= 0:
					print('z_in1 positive', z_in1)
					# update bias
					bias[a] = round(bias[a] + rate * (-1 - z_in1), 2)
					# update corresponding weights
					weights[a][a] = round(weights[a][a] + (rate * (-1 - z_in1) * x[i][a]), 2)
					weights[b][a] = round(weights[b][a] + (rate * (-1 - z_in1) * x[i][b]), 2)
				elif z_in2 >= 0: # just z_in2 is greater than 0
					print('z_in2 positive', z_in2)
					#update bias
					bias[b] = round(bias[b] + rate * (-1 - z_in2), 2)
					# update weights
					weights[a][b] = round(weights[a][b] + (rate * (-1 - z_in2) * x[i][a]), 2)
					weights[b][b] = round(weights[b][b] + (rate * (-1 - z_in2) * x[i][b]), 2)
				# Reset converged
				if converged > 0: 
					converged = 0

		# Check if we've gone through the number of training pairs
		if count == pairs:
			era += 1 # increment epoch we're on
			count = 0 # reset count
			print('Era: ', era)
		# Check if the era is greater than epochs so we can stop
		if era >= epochs:
			stop = True
			print('Training converged after {x} epochs'.format(x=epochs))
			# Write to output file
			with open(outfile, "w") as store:
				store.write("%d\n%d\n%d\n\n%s\n%s\n%.2f\n%.2f" %(inputs, outputs, pairs, weights, bias, v1, v2))
			store.close()
			return outfile
			break

"""
Activation Function for Madaline

args: 
	z_in -> input value of hidden layer
output: 
	1 if value is greater than or equal to 0
	-1 if value is less than 0

"""
def activateF(z_in):
	if z_in >= 0: return 1
	else: return -1

"""
Helper function to read in the .dat file and store the 
	data inside of it

arg: aFile - file to be read
"""
def readToStore(aFile):
	try:
		tests = []
			# Open and store contents of data file
		with open(aFile) as a_open:
			contents = a_open.readlines()
		# Read through specific lines to store input,outputs, pairs and training sets
		for i, line in enumerate(contents):
			if i == 0:
				t_ins = line
				t_ins = int(t_ins)
			elif i == 1:
				t_outs = line
				t_outs = int(t_outs)
			elif i == 2:
				t_pairs = line
				t_pairs = int(t_pairs)
			# Dealing with the samples
			else:
				x = line.strip()
				tests.append(x.split())
		return t_ins, t_outs, t_pairs, tests 
	except TypeError:
		print("Invalid File")
		os.system('clear')
		main()


"""
Function to gather the weights, epochs, rates and output file name 
	for Madaline NN 
"""
def initializeIt():
	try:
		weights = int(input("\nEnter '0' to initialize weights to zero, or enter '1'" 
			+ "to initialize weights to random values between -0.5 and 0.5:\n>> "))
		if weights not in range(0, 2):
			print("The value needs to be either a 0 or a 1\n")
			initializeIt()
		# Store number of user defined epochs
		epochs = int(input("\nEnter the max number of training epochs: \n>> "))
		# Get learning rate alpha
		rate = float(input("\nEnter learning rate alpha from 0 to 1 (not including 0):\n>>  "))
		if not 0 < rate <= 1:
			print("Your alpha value needs to be between 0 and 1, but not including 0.\n>>")
			initializeIt()
		# Get output file name
		outfile = input("Enter the file name where the weights will be saved: \n>>")
		if os.path.exists(outfile) and not os.stat(outfile).st_size == 0:
			print('Try again, that file is not empty') 
			initializeIt() 
		if not outfile.endswith('.txt'):
			print('You need to enter a text file')
			initializeIt()
	except KeyboardInterrupt:
		sys.exit()
	except ValueError:
		print("\nYou entered an incorrect value, please try again")
		time.sleep(1)
		os.system('clear')
		initializeIt()
	return weights, epochs, rate, outfile

def main():
	global weights, epochs, rate, outfile, ins, outs, pairs, tsets
	# Greet/get initial input file
	infile = mm.greetIn()
	contents = readToStore(infile)
	ins, outs, pairs, tsets = contents
	# Call menu 
	mm.menu()
	while(1):
		try:
			# Store choice and call cooresponding menu action
			choice = int(input(">>> "))
			if choice == 1:
				val = mm.pick_one(choice) 
				# Store values from initilizeIt()
				weights, epochs, rate, outfile = val
				t = Training(ins, outs, pairs, weights, tsets)
				# Store output file from training
				of = trainAlgo(t.tset, t.bias, rate, epochs, t.weights, t.pairs, t.inputs, outfile, t.outputs)
				"""
				Will need to call new prompt
				"""
				break
			mm.pick_one(choice)
			break
		except ValueError:
			print('Need to enter a number that matches one of the options')
			mm.menu()
		except KeyboardInterrupt:
			os.system('clear')
			mm.pick_one(3)

# Testing file
if __name__ == '__main__':
	main()


