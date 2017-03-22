"""
File for the madaline neural network

Ask user input for threshold (which is the error)
Check if weights have changed within that threshold then we say it's covnerged
"""
import madmenu as mm
import random as r
import os
import time
import sys

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
Function to calculate the change in the weights
"""
def upWeights(rate, one, zin, x):
	return round(rate * (one - zin) * x, 2)
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
	threshold: The value for checking if we have an error
"""
def trainAlgo(tsets, bias, rate, epochs, weights, pairs, inputs, outfile, threshold):
	# Hardcode value for  for v1, v2 bias
	v1 = v2 = bias[2] = .5
	x = [] # to store x inputs
	t = [] # to store t outputs
	stop = False # for determining convergence
	a, b, c = 0, 1, 2 # for the bias/weights
	count = 0 # keep track of number of inputs we've done so far
	era = 1 # Keep track of number of epochs
	temp = []
	converged = 0

	# Store each training pair, s:t 
	for i,tset in enumerate(tsets):
		if i % 3 == 1: # to get the inputs
			x.append(tset)
		if i % 3 == 2: # to get the outputs
			t.append(tset)
	while not stop:
		# Get the input to the hidden layer
		for i in range(pairs):
			print('\nThis weights for each round are: ', weights)
			print('This is epoch: ', era)
			count += 1
			z_in1 = round(bias[a] + (x[i][a] * weights[a][a]) + (x[i][b] * weights[b][a]), 2)
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
			print('This is t', t[i][a])
			
			# Check if error occured
			if t[i][a] == y: # Gets corresponding output
				print('t == y') 
				converged += 1 # Increment converged
				print('converged equals: ', converged)
				if converged == pairs:
					print('We have converged')
					stop = True
					print('Training converged after {x} epochs'.format(x=era))
					# Write to output file
					with open(outfile, "w") as store:
						store.write("%d\n%d\n\n%s\n%s\n%.2f\n%.2f" %(inputs, pairs, weights, bias, v1, v2))
					store.close()
					break
				continue
			# t = 1 and z1/z2 = -1
			elif t[i][a] == 1.0 and z1 == -1 or z2 == -1: # check if t = 1 and z1/z2 = -1
				val = min((z_in1, z_in2), key=lambda x: abs(x - 0)) # Get z value closest to 0
				# Get the possible changes
				cb1 = upWeights(rate, 1, z_in1, 1)
				cb2 = upWeights(rate, 1, z_in2, 1)
				czin1 = upWeights(rate, 1, z_in1, x[i][a])
				czin11 = upWeights(rate, 1, z_in1, x[i][b])
				czin2 = upWeights(rate, 1, z_in2, x[i][a])
				czin22 = upWeights(rate, 1, z_in2, x[i][b])
				# Both Z_in1/z_in2 are equal
				if val == z_in1 and val == z_in2: 
					pick = randint(1, 2)
					if pick == 1:
						print('Z_in1 and Z_in2 are equal, so update just z_in1')
						if czin1 < threshold and czin11 < threshold and cb1 < threshold:
							print('Error is small enough')
							converged += 1
							print('converged equals: ', converged)
							if converged == pairs:
								print('We have converged')
								stop = True
								print('Training converged after {x} epochs'.format(x=era))
								# Write to output file
								with open(outfile, "w") as store:
									store.write("%d\n%d\n\n%s\n%s\n%.2f\n%.2f" %(inputs, pairs, weights, bias, v1, v2))
								store.close()
							break
							continue
						else:
							print('This is what the z_in1 weights will change by: ', czin1, czin11)
							bias[a] = round(bias[a] + cb1)
							weights[a][a] = round(weights[a][a] + czin1, 2)
							weights[b][a] = round(weights[b][a] + czin11, 2)
							if converged > 0: converged = 0
							continue
					else: # choose 2
						print('zin1 and zin2 are equal, but update zin2')
						if czin2 < threshold and czin22 < threshold and cb2 < threshold:
							print('Error too small')
							converged += 1
							print('converged equals: ', converged)
							if converged == pairs:
								print('We have converged')
								stop = True
								print('Training converged after {x} epochs'.format(x=era))
								# Write to output file
								with open(outfile, "w") as store:
									store.write("%d\n%d\n\n%s\n%s\n%.2f\n%.2f" %(inputs, pairs, weights, bias, v1, v2))
								store.close()
								break
							continue
						else: 
							print('Zin1 Weights will change by: ', czin2, czin22)
							bias[b] = round(bias[b] + cb2)
							weights[a][b] = round(weights[a][b] + czin2, 2)
							weights[b][b] = round(weights[b][b] + czin22, 2)
							# reset converged
							if converged > 0: converged = 0
							continue
				if val == z_in1: # Both Z_in1/z_in2 are equal
					# Update bias and weight values of z_in1
					if czin1 < threshold and czin11 < threshold and cb1 < threshold:
						print('Error is small enough')
						converged += 1
						print('converged equals: ', converged)
						if converged == pairs:
							print('We have converged')
							stop = True
							print('Training converged after {x} epochs'.format(x=era))
							# Write to output file
							with open(outfile, "w") as store:
								store.write("%d\n%d\n\n%s\n%s\n%.2f\n%.2f" %(inputs, pairs, weights, bias, v1, v2))
							store.close()
							break
						continue
					else:
						print('This is what the zin1 weights will change by: ', czin1, czin11)
						weights[a][a] = round(weights[a][a] + czin1, 2)
						weights[b][a] = round(weights[b][a] + czin11, 2)
						# update bias
						bias[a] = round(bias[a] + cb1)
						if converged > 0: converged = 0
						continue
				if val == z_in2:
					# Update the weight and bias corresponding to z2
					if czin2 < threshold and czin22 < threshold and cb2 < threshold:
						print('Error is small enough')
						converged += 1
						print('converged equals: ', converged)
						if converged == pairs:
							print('We have converged')
							stop = True
							print('Training converged after {x} epochs'.format(x=era))
							# Write to output file
							with open(outfile, "w") as store:
								store.write("%d\n%d\n\n%s\n%s\n%.2f\n%.2f" %(inputs, pairs, weights, bias, v1, v2))
							store.close()
							break
						continue
					else:
						print('This is what the zin2 weights will change by: ', czin2, czin22)
						bias[b] = round(bias[b] + cb2)
						weights[a][b] = round(weights[a][b] + czin2, 2)
						weights[b][b] = round(weights[b][b] + czin22, 2)
						if converged > 0: converged = 0
						continue
			# t = -1 and z1 0r z2 = 1
			elif t[i][a] == -1.0: 
				# Get possible weight changes
				cb1 = upWeights(rate, -1, z_in1, 1)
				cb2 = upWeights(rate, -1, z_in2, 1)
				czin1 = upWeights(rate, -1, z_in1, x[i][a])
				czin11 = upWeights(rate, -1, z_in1, x[i][b])
				czin2 = upWeights(rate, -1, z_in2, x[i][a])
				czin22 = upWeights(rate, -1, z_in2, x[i][b])
				if z1 == 1 and z2 == 1:
				# Both z_in1 and z_in2 have positive inputs 
					if czin1 < threshold and czin11 < threshold and cb1 < threshold or czin2 < threshold and czin22 < threshold and cb2 < threshold:
						print('Error is small enough')
						converged += 1
						print('converged equals: ', converged)
						if converged == pairs:
							print('We have converged')
							stop = True
							print('Training converged after {x} epochs'.format(x=era))
							# Write to output file
							with open(outfile, "w") as store:
								store.write("%d\n%d\n\n%s\n%s\n%.2f\n%.2f" %(inputs, pairs, weights, bias, v1, v2))
							store.close()
							break
						continue
					else: # Not small enough
						# update weights
						weights[a][a] = round(weights[a][a] + czin1)
						weights[a][b] = round(weights[a][b] + czin2)
						weights[b][b] = round(weights[b][b] + czin22)
						weights[b][a] = round(weights[b][a] + czin11)
						# update bias
						bias[a] = round(bias[a] + cb1, 2)
						bias[b] = round(bias[b] + cb2, 2)
						if converged > 0: converged = 0
						continue
				# Only z_in1 is positive
				elif z_in1 >= 0:
					print('z_in1 positive', z_in1)
					if czin1 < threshold and czin11 < threshold:
						print('Error is small enough')
						converged += 1
						print('converged equals: ', converged)
						if converged == pairs:
							print('We have converged')
							stop = True
							print('Training converged after {x} epochs'.format(x=era))
							# Write to output file
							with open(outfile, "w") as store:
								store.write("%d\n%d\n\n%s\n%s\n%.2f\n%.2f" %(inputs, pairs, weights, bias, v1, v2))
							store.close()
							break
						continue
					else: # Not small enough
						# update weights
						weights[a][a] = round(weights[a][a] + czin1)
						weights[b][a] = round(weights[b][a] + czin11)
						# update bias
						bias[a] = round(bias[a] + cb1, 2)
						if converged > 0: converged = 0
						continue
				elif z_in2 >= 0: # just z_in2 is greater than 0
					print('z_in2 positive', z_in2)
					if czin2 < threshold and czin22 < threshold and cb2 < threshold:
						print('Error is small enough')
						converged += 1
						print('converged equals: ', converged)
						if converged == pairs:
							print('We have converged')
							stop = True
							print('Training converged after {x} epochs'.format(x=era))
							# Write to output file
							with open(outfile, "w") as store:
								store.write("%d\n%d\n\n%s\n%s\n%.2f\n%.2f" %(inputs, pairs, weights, bias, v1, v2))
							store.close()
							break
						continue
					else: # not small enough
						weights[a][b] = round(weights[a][b] + czin2)
						weights[b][b] = round(weights[b][b] + czin22)
						# update bias
						bias[b] = round(bias[b] + cb2, 2)
						if converged > 0: converged = 0
						continue
		# Check if we've gone through the number of training pairs
		if count == pairs:
			era += 1 # increment epoch we're on
			count = 0 # reset count
		# Check if the era is greater than epochs so we can stop
		if era >= epochs:
			stop = True
			print('Training converged after {x} epochs'.format(x=era))
			# Write to output file
			with open(outfile, "w") as store:
				store.write("%d\n%d\n\n%s\n%s\n%.2f\n%.2f" %(inputs, pairs, weights, bias, v1, v2))
			store.close()
			return outfile
			
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
Helper function to read in the .txt file and store the 
	data inside of it (only works for training text file)

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
Get the weights for the samples
"""
def gimmeWeights():
	try:
		# Get/store weights
		weights = int(input("\nEnter '0' to initialize weights to zero, or enter '1'" 
			+ "to initialize weights to random values between -0.5 and 0.5:\n>> "))
		if weights not in range(0, 2):
			print("The value needs to be either a 0 or a 1\n")
			initializeIt()
		return weights
	except ValueError:
		print('The value needs to be either a 0 or a 1')
		gimmeWeights()
"""
Get number of epochs for training the NN
"""
def howLong():
	try:
		# Store number of user defined epochs
		epochs = int(input("\nEnter the max number of training epochs: \n>> "))
		if epochs <= 0:
			print('The number of epochs need to be greater than 0')
			howLong()
		return epochs
	except ValueError:
		print('The number of epochs need to be a whole number, greater than 0')
		howLong()

"""
Function to get the learning rate for NN
"""
def learnRate():
	try:
		# Get learning rate alpha
		rate = float(input("\nEnter learning rate alpha from 0 to 1 (not including 0):\n>>  "))
		if not 0 < rate <= 1:
			print("Your alpha value needs to be between 0 and 1, but not including 0.\n>>")
			learnRate()
	except ValueError:
		print('The learning rate needs to be a decimal between 0 (but not including 0) and 1')
		learnRate()
	return rate

"""
Function to get the output file for either the training or testing file
"""
def storeMe():
	try:
		# Get output file name for results
		outfile = input("\nEnter the file name where the results will be saved: \n>>")
		if os.path.exists(outfile) and not os.stat(outfile).st_size == 0:
			print('Try again, that file is not empty') 
			storeMe() 
		if not outfile.endswith('.txt'):
			print('You need to enter a text file')
			storeMe()
	except ValueError:
		print('I need a text file that is empty mija')
		storeMe()
	return outfile

"""
Function to get the testing file for the NN
"""
def getMe():
	try:
		# Get testing file for NN
		tFile = input("\nEnter the file name of the inputs you'd like to test: \n>>")
		# Check if file doesn't exist or it's empty
		if not os.path.exists(tFile) or not os.path.getsize(tFile) > 0:
			print('The file entered needs to be a non-empty file')
			getMe()
		# File doesn't end with .txt
		if not tFile.endswith('.txt'):
			print('The file needs to be a text file')
			getMe()
	except ValueError:
		print('Please enter a non-empty testing file')
		getMe()
	return tFile

"""
Function to get the threshold from the user for checking when we've converged
"""
def errorDiff():
	try:
		threshold = float(input('\nPlease enter a threshold to check for convergence: \n>> '))
		if threshold > 0.1:
			print('Please choose a threshold between 0 and 0.1')
			errorDiff()
	except ValueError():
		print('Please enter a decimal between 0 and 0.1')
		errorDiff()
	return threshold

"""
Function to gather the weights, epochs, rates and output file name 
	for training Madaline NN 
"""
def initializeIt():
	try:
		# Get and store weights, epochs, rates and the name of the output file
		weights = gimmeWeights()
		epochs = howLong()
		threshold = errorDiff()
		rate = learnRate()
		outfile = storeMe()
	except KeyboardInterrupt:
		sys.exit()
	return weights, epochs, threshold, rate, outfile

"""
Function to read and store the values of testing file

arg: 
	testfile - file to read and store the inputs and training sets
"""
def read2Store(testfile):
	try:
		tests = []
			# Open and store contents of data file
		with open(testfile) as t_open:
			contents = t_open.readlines()
		# Read through specific lines to store input, pairs and testing sets
		for i, line in enumerate(contents):
			if i == 0:
				t_ins = line
				t_ins = int(t_ins)
			elif i == 1:
				t_pairs = line
				t_pairs = int(t_pairs)
			# Dealing with the samples
			else:
				x = line.strip()
				tests.append(x.split())
	except TypeError:
		print("Invalid File")
		os.system('clear')
		mm.menu()
	return t_ins, t_pairs, tests

"""
Function to get both the testing file and the trained NN weights
"""
def testIt():
	try:
		# Gets the test file and the inputs, pairs and tsetset
		tstfile = getMe()
		ins, pairs, tstset = read2Store(tstfile)
		# Get empty output file for the results
		outfile = storeMe()
	except KeyboardInterrupt:
		sys.exit()
	return outfile, ins, pairs, tstset
	# return tstfile, outfile
"""
Function to test the trained NN

arg: 
	weights - weights from the trained NN
	tstset - set of inputs to test
	bias - weights of the bias' from the trained NN
	v1/v2 - the weights going into the output layer from the hidden layer
"""
def testNN(weights, tstset, bias, v1, v2):
	x = []
	# Store each training pair, s:t 
	for tset in tstset:
		if tset != '':
			x.append(tset)
	



def main():
	# Greet/get initial input file
	infile = mm.greetIn()
	contents = readToStore(infile)
	ins, outs, pairs, tsets = contents
	# To make the menu repeat each time
	while(1):
		# Call menu 
		mm.menu()
		try:
			# Store choice and call cooresponding menu action
			choice = int(input(">>> "))
			if choice == 1:
				val = mm.pick_one(choice) 
				# Store values from initilizeIt()
				weights, epochs, threshold, rate, outfile = val
				t = Training(ins, outs, pairs, weights, tsets)
				# Store output file from training
				of = trainAlgo(t.tset, t.bias, rate, epochs, t.weights, t.pairs, t.inputs, outfile, threshold)
				continue
			# Testing the NN
			if choice == 2: 
				# Store values from testIt
				val = mm.pick_one(choice)
				ofile, ins, pairs, tstset = val
				print('I get here')
				testNN(0, tstset, 0, 0, 0)
				#continue
				# outF = storeMe() # Get output file to store the results
			# mm.pick_one(choice)
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


