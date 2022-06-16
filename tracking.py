from my_utils import *


def parse_outputs(layerOutputs, video_width, video_height):
    """
    :param layerOutputs: outputs from the 3 different classification layers of YOLO
    :param video_width: frame size
    :param video_height: frame size
    :return:(boxes, confidences, classids): 3 lists of coordinates, probabilities and classes
    of the final retained objects
    """
    boxes, confidences, classids = [], [], []
    for output in layerOutputs:
        for i, detection in enumerate(output):
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > preDefinedConfidence:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Printing the info of the detection
                # print('\nName:\t', LABELS[class_id],
                # '\t|\tBOX:\t', x,y)

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classids.append(class_id)
    return boxes, confidences, classids


def count_vehicles(ids, boxes, class_ids, vehicle_count, previous_frame_detections):
    """
    This function isn't used. Use the lane.count feature instead, drawing an area over entire
    image if necessary
    :param ids:
    :param boxes:
    :param class_ids:
    :param vehicle_count:
    :param previous_frame_detections:
    :return:
    """
    current_detections = {}
    curr_vehicle_count = 0
    # ensure at least one detection exists
    if len(ids) > 0:
        # loop over the indices we are keeping
        for i in ids.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            center_x = x + (w // 2)
            center_y = y + (h // 2)

            # When the detection is in the list of vehicles, AND
            # it crosses the line AND
            # the ID of the detection is not present in the vehicles
            if LABELS[class_ids[i]] in objects_of_interest:
                current_detections[(center_x, center_y)] = curr_vehicle_count
                # curr_vehicle_count +=1
        labels, vehicle_count, current_detections = match_labels(previous_frame_detections,
                                                                 current_detections,
                                                                 vehicle_count)

        for (x, y) in current_detections.keys():
            vehicle_id = current_detections.get((x, y))
            # cv2.putText(frame, str(vehicle_id), (x, y), \
            #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)

    return vehicle_count, current_detections


def dist(x1, x2):
    """I like geometry"""
    return np.sum((x1-x2)**2, axis = 1)


def single_dist(x1, x2):
    """I like geometry"""
    return np.sum((x1-x2)**2)


def tiebreak(prev_labels, centers, labels, vehicle_count, current_detections):
    """

    :param prev_labels: vehicle labels/coordinates from previous frame
    :param centers: coordinates of detected vehicles in current frame
    :param labels: I don't remember
    :param vehicle_count:
    :param current_detections:
    :return:
    """
    for label in set(labels):
        ixs = list(np.where(np.array(labels) == label)[0])
        if len(ixs) > 1:
            distances = [single_dist(np.array(prev_labels[label]), np.array(centers[ix])) for ix in ixs]
            closest = np.argmin(distances)
            labels[ixs[closest]] = label
            ixs.pop(closest)
            for ix in ixs:
                labels[ix] = vehicle_count
                current_detections[tuple(centers[ix])] = vehicle_count
                vehicle_count += 1
    return labels, vehicle_count, current_detections


def match_labels(previous_detections, current_detections, vehicle_count):
    """

    :param previous_detections:
    :param current_detections:
    :param vehicle_count:
    :return:
    """
    most_recent_positions = get_most_recent_position(previous_detections)
    prev_centers = np.array(list(most_recent_positions.keys()))
    centers = np.array(list(current_detections.keys()))
    labels = []
    for center in centers:
        closest_i = np.argmax(-1*dist(center, prev_centers))
        closest = prev_centers[closest_i]
        labels.append(most_recent_positions[(closest[0], closest[1])])
        current_detections[(center[0], center[1])] = most_recent_positions[tuple(prev_centers[closest_i])]
    prev_labels = {label:center for center, label in most_recent_positions.items()}
    labels, vehicle_count, current_detections = tiebreak(prev_labels, centers, labels, vehicle_count, current_detections)
    return labels, vehicle_count, current_detections


def get_most_recent_position(previous_frame_detections):
    most_recent_positions = {}
    for detections in previous_frame_detections:
        prev_labels = {label:center for center, label in detections.items()}
        for label in prev_labels.keys():
            most_recent_positions[label] = prev_labels[label]
    most_recent_positions = {label:center for center, label in most_recent_positions.items()}
    return most_recent_positions


def identify_vehicles(boxes,
                      confidences,
                      classids,
                      frame,
                      vehicle_count,
                      all_detections):
    ids = cv2.dnn.NMSBoxes(boxes, confidences, preDefinedConfidence,
                            preDefinedThreshold)
    draw_boxes(ids, boxes, classids, confidences, frame)
    previous_detections = all_detections[-FRAMES_BEFORE_CURRENT:]
    vehicle_count, current_detections = count_vehicles(ids,
                                                       boxes,
                                                       classids,
                                                       vehicle_count,
                                                       previous_detections)
    current_positions = {label: center for center, label in current_detections.items()}
    return ids, vehicle_count, current_detections, current_positions
