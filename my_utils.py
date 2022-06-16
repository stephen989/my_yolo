import cv2
import numpy as np
from config import *
import matplotlib.path as mpltPath
import pickle
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt


np.random.seed(1)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")


def get_eqn(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    A = (y2-y1)/(x2-x1)
    B = -1
    c = y1-A*x1
    return A, B, c


def distance_to_line(A, B, c, px, py):
    return abs(A*px + B*py + c)/np.sqrt(A**2 + B**2)


class Lane:
    """Class for lanes/polygons"""
    def __init__(self, points, name="Lane"):
        self.points = points.reshape(len(points), 2)
        self.vehicles = 0
        self.name = name
        self.path = mpltPath.Path(self.points)
        self.count = 0
        self.history = dict()
        self.counts = list()

    def draw(self, frame, color=(0, 0, 0)):
        color = np.random.randint(0, 255, 3)
        color = (0,170,255)
        color = (200,200,200)
        frame = cv2.polylines(frame,
                              [self.points.astype(np.int32)],
                              isClosed=True,
                              color=color,
                              thickness=2)
        return frame

    def count_vehicles(self, current_positions):
        coords = list(current_positions.values())
        ids = np.array(list(current_positions.keys()))
        is_in = self.path.contains_points(coords)
        vehicle_ids = ids[is_in]
        self.history[len(self.history.keys())] = vehicle_ids
        self.count = sum(is_in)
        self.counts.append(self.count)
        return self.count

    def label(self, frame):
        """Draws ugly label of lane name on frame"""
        y = 0.5 * (self.points.max(0)[1] + self.points.min(0)[1])
        x = 0.5 * (self.points.min(0)[0] + self.points.max(0)[0])
        write_text(frame, self.name, (x, y), anchor="center")


class Line:
    """Line class"""
    def __init__(self, points, name="Line"):
        self.point1, self.point2 = points.reshape(2, 2)
        self.points = points.reshape(2, 2)
        x1_temp, y1_temp = self.point1
        x2_temp, y2_temp = self.point2
        self.x1 = min(x1_temp, x2_temp)
        self.x2 = max(x1_temp, x2_temp)
        self.y1 = min(y1_temp, y2_temp)
        self.y2 = max(y1_temp, y2_temp)
        self.A, self.B, self.c = get_eqn(self.point1, self.point2)
        self.crossings = []
        self.name = name

    def draw(self, image, color = (0,0,0)):
        color = (51,85,255)
        cv2.line(image,
                 tuple(self.point1.astype(np.int32)),
                 tuple(self.point2.astype(np.int32)),
                 color,
                 2)

    def detect_crossings(self, current_positions):
        new_crossings = []
        for vehicle in current_positions.keys():
            position = current_positions[vehicle]
            if (position[0] + horizontal_margin > self.x1) and ((position[0] - horizontal_margin < self.x2)):
                if (position[1] + vertical_margin > self.y1) and ((position[1] - vertical_margin < self.y2)):
                    if distance_to_line(self.A, self.B, self.c, position[0], position[1]) < distance_margin:
                        new_crossings.append(vehicle)
        new_crossings = [crossing for crossing in new_crossings if crossing not in self.crossings]
        # if new_crossings:
        #     print(f"{self.name} crossings: ", " ".join([str(vehicle) for vehicle in new_crossings]))
        self.crossings += new_crossings

    def label(self, frame):
        y = np.mean(self.points[:,1]) + 40
        x = np.min(self.points[:,0])
        write_text(frame, self.name, (x, y))


def draw_dashcam_boxes(boxes, classIDs, confidences, frame):
    """For drawing boxes in dashcam.py only"""
    color = (250,250,30)
    ids = cv2.dnn.NMSBoxes(boxes, confidences, preDefinedConfidence,
                            preDefinedThreshold)
    counts = {name: 0 for name in LABELS}
    # ensure at least one detection exists
    if len(ids) > 0:
        # loop over the indices we are keeping
        for i in ids.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            # color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            classname = LABELS[classIDs[i]]
            if classname in objects_of_interest:
                counts[classname] += 1
                text = "{}: {:.2f}".format(classname, confidences[i])
                cv2.putText(frame, text.upper(), (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # Draw a green dot in the middle of the box
                cv2.circle(frame, (x + (w // 2), y + (h // 2)), 2, (0, 0xFF, 0), thickness=2)
        text = "\n".join([f"{vehicle}: {str(counts[vehicle])}" for vehicle in objects_of_interest])
        write_text(frame, text)


def draw_boxes(ids, boxes, classIDs, confidences, frame):
    """Draws boxes retained by NMS on frame with class, id and confidence written"""
    color = (205, 222, 239)
    color = (75, 10, 10)
    # ensure at least one detection exists
    if len(ids) > 0:
        # loop over the indices we are keeping
        for i in ids.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            # color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h),  color, 1)
            text = "{}: {:.2f}".format(LABELS[classIDs[i]],
                                       confidences[i])
            cv2.putText(frame, text.upper(), (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Draw a green dot in the middle of the box
            cv2.circle(frame, (x + (w // 2), y + (h // 2)), 2, (0, 0xFF, 0), thickness=2)


def write_text(frame,
               lines,
               position = (20, 20),
               anchor = "left",
               padding = 5,
               bg=(255,255,255)):
    """Creates text to be written on frame, formats it into white rectangle and writes to frame
        Ugly function but opencv is the worst"""
    lines = lines.split("\n")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    font_color = (0, 0, 0)
    font_thickness = 8

    text_sizes = [cv2.getTextSize(text, font, font_scale, font_thickness)[0] for text in lines]
    total_height = sum([height for (width, height) in text_sizes]) + padding * (len(lines) + 1)
    total_height += int(total_height/len(lines))
    total_width = max([width for (width, height) in text_sizes]) + padding + 50
    if anchor == "left":
        x, y = position
    else:
        x, y = position
        x -= 0.75*total_width
    x, y = int(x), int(y)
    cv2.rectangle(frame, (x, y + total_height), (x + total_width, y-padding), bg, -1)
    padding = 10
    if "" in lines:
        lines.remove("")
    for text in lines:
        left, right = text.split(":")
        # new_text = f"{left[:9]+':': <10}{right: >4}"
        new_text = f"{left}: {right}"
        text_size, _ = cv2.getTextSize(new_text, font, font_scale, font_thickness)
        text_w, text_h = text_size
#         cv2.rectangle(frame, (x, y-text_h), (x + text_w, y - text_h), bg, -1)
        cv2.putText(frame,
                    new_text.upper(),
                    (x, y+text_h),
                    font,
                    font_scale,
                    font_color)
        y += text_h + padding


def create_video_writer(video_width, video_height, video_stream, output_path):
    """Creates video writer"""
    # Getting the fps of the source video
    video_fps = video_stream.get(cv2.CAP_PROP_FPS)
    # initialize our video writer
    fourcc = cv2.VideoWriter_fourcc(*"MPEG")
    return cv2.VideoWriter(output_path, fourcc, video_fps,
                           (video_width, video_height), True)


def get_frame_text(lanes, lines, fps):
    frame_text = ""
    for line in lines:
        frame_text += f"\n{line.name}: {len(line.crossings)}"
    for lane in lanes:
        frame_text += f"\n{lane.name}: {lane.count}"
    nlane_switches = sum([len(line.crossings) for line in lines if "Lane" in line.name])
    # frame_text += f"\nLane Switches: {nlane_switches}"
    frame_text += f"\nFPS: {fps:.2f}"
    return frame_text


def pickle_dump(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def create_lane_log(lanes, max_id):
    log = np.zeros((len(lanes), len(lanes[0].history), max_id)).astype("int")
    for j, lane in enumerate(lanes):
        for i in range(len(lanes[0].history)):
            cars = lane.history[i]
            for car in cars:
                try:
                    log[j, i, car] = 1
                except Exception as e:
                    pass
    return log


def get_switches(log):
    """Detect lane switches from aggregate lane log created from create_lane_log"""
    switches = []
    for car in range(log.shape[2]):
        lanes_visited = np.where(log.sum(1)[:,car]>3)[0]
        nlane = len(lanes_visited)
        if nlane > 1:
            times = np.zeros((3, 2), np.int32)
            for lane in lanes_visited:
                arr = sliding_window_view(log[lane,:,car], window_shape=5).max(1)
                in_lane = np.where(arr==1)[0]
                joined_lane = in_lane[0]
                left_lane = in_lane[-1]
                times[lane] = [joined_lane, left_lane]
            times[np.where(times==0)[0]] = 1e6
            current_lane = np.argmin(times[:,0])
            times[current_lane,0] = 1e6
            while np.min(times[:,0]) < 1e6:
                next_lane = np.argmin(times[:,0])
                switches.append((car, current_lane, next_lane, times[current_lane, 1]))

                current_lane = next_lane
                times[current_lane,0] = 1e6
    return switches


def get_counts(lanes, df):
    counts = np.zeros((len(df), len(lanes)))
    for i in range(len(df)):
        counts[i] = [count(lane, df.iloc[i].dropna().values.tolist()) for lane in lanes]
    return counts


def count(lane, vals_list):
    return lane.path.contains_points(vals_list).sum()


def create_plot_img(fig, counts):
    plt.style.use('fivethirtyeight')
    plt.xlim((0, len(counts)))
    plt.ylim((0, 1.25 * np.max(counts)))
    plt.plot(counts)
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                        sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig.axes[0].set_prop_cycle(color=['red', 'green', 'blue'])
    return fig, img