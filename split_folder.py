import splitfolders

input_folder = 'C:/Users/91909/Documents/My_Projects/Machine_Learning_&_AI/deep_learning/Rock_Paper_Scissors/Rock_Paper_Scissors_Dataset'
output_folder = 'C:/Users/91909/Documents/My_Projects/Machine_Learning_&_AI/deep_learning/Rock_Paper_Scissors/Train_Test_Validation'

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
# Train, validation and test split

splitfolders.ratio(input_folder, output=output_folder, 
                   seed=42, ratio=(.7, .2, .1), 
                   group_prefix=None)

# Split val/test with a fixed number of items e.g. 100 for each set.
# To only split into training and validation set, use a single number to `fixed`, i.e., `10`.
# enable oversampling of imbalanced datasets, works only with fixed
# splitfolders.fixed(input_folder, output="cell_images2", 
#                    seed=42, fixed=(35, 20), 
#                    oversample=False, group_prefix=None)