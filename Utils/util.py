import random
import string


def random_name_generator(length=20):
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for i in range(length))


##### Check cars in the frame/image and return list of the car ######
##### Why we need this function? because there are multiplr cars in a single frame/image ######
def find_car_indexes(lst):
    indices = []
    for i, elem in enumerate(lst):
        if elem == 2:##### 3l is for car label
            indices.append(i)
    return indices

##### Calculate the recatangle area #####
def calculate_area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

#####  This function find the biggest car(only one) and returns its dimension ######
#####  Why need this function? because frame/image have multiple cars we need only one user cars\
#####  the car is very closed to th camera is the users car and the car is closed to the camers has largest\
#####  area thats why we calculate area of the cars and find biggest area of the car because that one is users car. #####
def find_biggest_car_bbox(bboxes):
    biggest_bbox = None
    max_area = 0
    for bbox_dict in bboxes:
        bbox = bbox_dict["bbox"]  
        area = calculate_area(bbox)  # calculate car rectangle(bbox) area
        if area > max_area:
            max_area = area
            biggest_bbox = bbox_dict  # Store the entire dictionary
    return biggest_bbox


