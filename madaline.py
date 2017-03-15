"""
File for the madaline neural network
"""
import madmenu as mm

"""
class for Training Net
"""
class Training(object):
	def __init__(self, inputs, outputs, pairs, weights, tsets):
		self.inputs = inputs
		self.outputs = outputs
		self.pairs = pairs

		# Set inputs and their weights
        # Set initial weight to random value
        if weights == 1: 
            val = float(random.uniform(-0.5, 0.5))
            # Round to tenth of decimal
            self.inweight = round(val, 2)
        # Set initial weight to 0
        else: self.inweight = weights
        # Create matrix to store w[i][j] values of weights
        self.weights = [[self.inweight for x in range(self.num_outputs)] for y in range(self.num_inputs)]



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
		mm.greetIn()


"""
Function to gather the weights, epochs, rates and output file name 
	for Madaline NN 
"""
def initializeIt():
	# global weights, epochs, rate, outfile
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
				break
			mm.pick_one(choice)
			break
		except ValueError:
			print('Need to enter a number that matches one of the options')
			mm.menu()

# Testing file
if __name__ == '__main__':
	main()

