from sklearn.model_selection import train_test_split
from my_functions import split_data, order_test_set


path_to_data = "C:\\Users\\User\\OneDrive\\Documents\\VSCode\\Python Projects\\Machine Learning and Neural Nets\\computer vision\\Train"
path_to_save_train = "C:\\Users\\User\\OneDrive\\Documents\\VSCode\\Python Projects\\Machine Learning and Neural Nets\\computer vision\\traffic_signs\\train"
path_to_save_val = "C:\\Users\\User\\OneDrive\\Documents\\VSCode\\Python Projects\\Machine Learning and Neural Nets\\computer vision\\traffic_signs\\val"    
split_data(path_to_data, path_to_save_train, path_to_save_val)

if False:
    path_to_images = "C:\\Users\\User\\OneDrive\\Documents\\VSCode\\Python Projects\\Machine Learning and Neural Nets\\computer vision\\Test"
    path_to_csv = "C:\\Users\\User\\OneDrive\\Documents\\VSCode\\Python Projects\\Machine Learning and Neural Nets\\computer vision\\Test.csv"

    order_test_set(path_to_images,path_to_csv)