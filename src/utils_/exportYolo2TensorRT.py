from ultralytics import YOLO

# Load a model
model = YOLO('/home/pqbas/catkin_ws/src/blueberry/src/detection/weights/22Sep23/yolov8m_best.pt')  # load an official model

# Export the model
model.export(format='engine')