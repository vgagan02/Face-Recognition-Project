import face_recognition
import os
import heapq
import cv2

class celebnode:
    def __init__(self, name, encoding=None):
        self.name = name
        self.encoding = encoding
        self.children = []

def loadimg(directory):
    celebrity_images = {}
    for celebrity in os.listdir(directory):
        celebrity_path = os.path.join(directory, celebrity)
        if os.path.isdir(celebrity_path):
            images = [os.path.join(celebrity_path, img) for img in os.listdir(celebrity_path)]
            celebrity_images[celebrity] = images
    return celebrity_images

def trainfn(train_directory):
    celebrity_images = loadimg(train_directory)
    celebrity_nodes = []

    for celebrity, images in celebrity_images.items():
        for img_path in images:
            image = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                celebrity_node = celebnode(name=celebrity, encoding=encoding[0])
                celebrity_nodes.append(celebrity_node)

    return celebrity_nodes

def similarity(encoding1, encoding2):
    return face_recognition.face_distance([encoding1], encoding2)[0]

def heuristicval(node, target_encoding):
    if node.encoding is not None:
        return similarity(node.encoding, target_encoding)
    else:
        return float('inf')

def astar(root, target_encoding):
    heap = [(0, 0, root)]  
    while heap:
        total_cost, _, current_node = heapq.heappop(heap)

        if current_node.encoding is not None and similarity(current_node.encoding, target_encoding) < 0.5:
            return current_node

        for child in current_node.children:
            cost = total_cost + 1
            priority = cost + heuristicval(child, target_encoding)
            heapq.heappush(heap, (cost, priority, child))

    return None

def treebuild(celebrity_nodes):
    root_node = celebnode(name="Root")
    for celebrity_node in celebrity_nodes:
        root_node.children.append(celebrity_node)

    return root_node

def visulaize(input_image, face_location, celebrity_name):
    top, right, bottom, left = face_location
    cv2.rectangle(input_image, (left, top), (right, bottom), (0, 0, 255), 2)

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(input_image, celebrity_name, (left + 6, bottom - 6), font, 0.5, (0, 0, 255), 1)

    cv2.imshow('Celebrity Recognition', input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


train_directory = r"D:\aiproj\data\train"
input_image_path = r"D:\aiproj\will.jpg"

celebrity_nodes = trainfn(train_directory)
data_tree = treebuild(celebrity_nodes)

input_image = face_recognition.load_image_file(input_image_path)
input_encoding = face_recognition.face_encodings(input_image)[0]

target_node = astar(data_tree, input_encoding)

if target_node:
    print(f"Found: {target_node.name}")
    visulaize(input_image, face_recognition.face_locations(input_image)[0], target_node.name)
else:
    print("Celebrity not found.")
