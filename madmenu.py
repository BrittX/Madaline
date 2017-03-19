"""
Menu for Madline Neural Network
"""
import os
import sys
import time
import madaline as m

"""
Function to create the interactive menu for the user
"""
def menu():
    #os.system('clear')

    print("\nPlease choose from the options below:")

    print("1. Train using a training data file")
    print("2. Test/Deploy Neural Net")
    print("3. Exit\n")

    return

"""
Function that calls the selected method to run.

Arg:
    choice - integer value for menu option
"""
def pick_one(choice):
    try:
        if choice in menu_choice:
            vals = menu_choice[choice]()
    # User entered number that's not valid
    except ValueError:
        print("Error: Invalid selection")
        time.sleep(1)
        menu()
    except KeyboardInterrupt: 
        os.system('clear')
        menu_choice[3]()
    # print(vals)
    return vals

"""
Print initial greeting and get input file
"""
def greetIn():
    print('Welcome to my Madaline Neural Net!\n')
    try: 
        # store as a input file
        infile = input('Enter the data input file name: ')
        if os.path.isfile(infile) and infile.endswith('.txt'): # return True if it is a file
            return infile
    except KeyboardInterrupt:
            os.system('clear')
            menu_choice[3]()
    '''
    except Error: # file not real or not a .txt file
            print('The file you entered does not exist.')
            print('Please enter a txt file')
            time.sleep(1)
            os.system('clear')
            greetIn()
    '''

# Options for menu
menu_choice = {"menu": menu,
        1: m.initializeIt, # Call main from madaline.py
        # 2: train_weight,
        #3: file_test,
        3: sys.exit
        }
