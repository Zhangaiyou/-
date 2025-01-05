import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageGrab, ImageTk
import numpy as np
import cv2
from keras.models import load_model
import os

# 加载训练好的模型
model = load_model('mnist_cnn_model.keras')


def process_image(image):
    """
    对给定图像进行格式转换。
    """
    # 缩小图像至(28, 28)，使用面积插值
    # 获取图像尺寸
    height, width = image.shape[:2]

    # 计算目标正方形边长，取长边
    side = max(height, width)

    # 创建正方形空白画布，用于居中放置原图
    square_image = np.zeros((side, side), dtype=image.dtype)

    # 计算居中放置的位置
    top = (side - height) // 2
    left = (side - width) // 2

    # 将原图放置到正方形画布的中心
    square_image[top:top + height, left:left + width] = image

    # 缩小到28x28
    img_resized = cv2.resize(square_image, (28, 28), interpolation=cv2.INTER_AREA)

    # 求质心并进行平移变换
    moments = cv2.moments(img_resized)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        # 中心坐标
        center_x, center_y = 14, 14

        # 水平和竖直像素距离
        dx = center_x - cx
        dy = center_y - cy

        # 平移变换矩阵
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img_centered = cv2.warpAffine(img_resized, M, (28, 28))
    else:
        img_centered = img_resized  # 如果图像为空，则不进行平移变换

    # 增加通道维度
    img_centered = img_centered[..., np.newaxis]

    # 增加批量维度
    img_centered = np.expand_dims(img_centered, axis=0)

    return img_centered


def analyze_connected_components(image):
    """
    对给定的图像进行连通组件分析，并返回最大连通组件的外接矩形区域。
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

    # 初始化最大面积和对应的组件索引
    max_area = 0
    max_label_index = -1

    # 遍历每个连通组件寻找最大面积的组件
    for i in range(1, num_labels):  # 跳过背景标签（索引为0）
        x, y, w, h, area = stats[i]
        if area > max_area:
            max_area = area
            max_label_index = i

    # 如果找到连通组件，提取其外接矩形区域
    if max_label_index != -1:
        x, y, w, h, _ = stats[max_label_index]
        penultimate_image = image[y:y + h, x:x + w]
        final_image = process_image(penultimate_image)
        return final_image  # 返回最大连通组件
    else:
        print("没有找到数字")
        return None  # 或者返回一个特定的值/None来表示没有找到组件


def vertical_projection(binary):
    """
    计算图像的垂直投影并进行。
    """
    # 计算垂直投影
    projection = np.sum(binary, axis=0)

    # 对投影数据进行平滑处理，以获得更平滑的线条
    # 这里使用简单的一维卷积，核大小为5，你可以根据需要调整
    smoothed_projection = np.convolve(projection, np.ones(5) / 5, mode='valid')

    # 二值化垂直投影
    binary_projection = np.where(smoothed_projection > 6000, 255, 0).astype(np.uint8)

    # 初始化变量以追踪跳变位置
    transition_points = []  # 用来存储所有跳变点
    in_black_region = False  # 标记是否处于黑色区域

    for ii, val in enumerate(binary_projection):
        if val == 255 and not in_black_region:  # 从白到黑的跳变
            transition_points.append(ii)
            in_black_region = True
        elif val == 0 and in_black_region:  # 从黑到白的跳变
            transition_points.append(ii)
            in_black_region = False

    # 初始化分割结果列表
    segmented_digits = []
    final_images = []

    if len(transition_points) >= 4:  # 至少需要两对跳变点来定义中间位置

        # 初始化中间点
        mid_point1 = 0

        for ii in range(1, len(transition_points) - 1, 2):

            # 计算中间点
            mid_point2 = (transition_points[ii] + transition_points[ii + 1]) // 2

            # 确保不会因为计算导致索引越界
            segment = binary[:, max(mid_point1, 0):min(mid_point2, binary.shape[1])]
            if segment.shape[1] > 200:  # 确保分割出的图像有实际宽度
                segmented_digits.append(segment)

            # 计算中间点
            mid_point1 = mid_point2

        if mid_point1 < binary.shape[1]:
            segment = binary[:, mid_point1:]
            if segment.shape[1] > 200:
                segmented_digits.append(segment)

    else:
        segment = binary
        if segment.shape[1] > 200:
            segmented_digits.append(segment)

    # 对分割出的每个数字进行连通组件分析
    for resized_digit in segmented_digits:
        # 确保图像不是空的
        if resized_digit is not None and resized_digit.size > 0:
            digit_with_max_component = analyze_connected_components(resized_digit)
            if digit_with_max_component is not None:
                final_images.append(digit_with_max_component)

    return final_images


def predict_digits(segmented_digits):
    """
    使用模型预测图像中的数字及其置信度
    """
    predictions = []
    for digit in segmented_digits:
        # 使用模型进行预测
        prediction = model.predict(digit)

        # 获取最可能的类别及其置信度
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100  # 置信度
        predictions.append((predicted_class, confidence))
    return predictions


def image_process_image(image_path):
    """
    对图像输入的图片进行预处理。
    """
    # 读取图像并灰度化
    img = Image.open(image_path)
    img_gray = img.convert('L')
    img_np = np.array(img_gray)

    # 二值化
    binary1 = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                    blockSize=81,
                                    C=2)

    # 使用中值滤波进行降噪
    denoised_img = cv2.medianBlur(src=img_np, ksize=37)


    # 二值化
    binary = cv2.adaptiveThreshold(denoised_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                   blockSize=81,
                                   C=2)

    final_image = vertical_projection(binary)

    return final_image


def handwriting_process_image(image_path):
    """
    对手写输入的图片进行预处理。
    """
    # 读取图像并灰度化
    img = Image.open(image_path)
    img_gray = img.convert('L')
    img_gray_np = np.array(img_gray)

    # 二值化
    binary = cv2.adaptiveThreshold(img_gray_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                   blockSize=81,
                                   C=2)

    final_image = vertical_projection(binary)

    return final_image


def main_process_image(image_path):
    """
    对成绩输入的试卷进行预处理。
    """
    img = cv2.imread(image_path)

    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义红色在HSV空间的阈值范围
    lower_red = np.array([0, 0, 0])  # 注意调整这些值以匹配你的图像中的红色
    upper_red = np.array([10, 255, 255])

    # 创建红色掩码
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # 应用掩码到原图
    red_only = cv2.bitwise_and(img, img, mask=mask)

    # 转回BGRY空间进行阈值处理
    gray = cv2.cvtColor(red_only, cv2.COLOR_BGR2GRAY)

    # 二值化
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                   blockSize=17,
                                   C=2)

    img_np = np.array(binary)

    # 使用中值滤波进行降噪
    denoised_img = cv2.medianBlur(src=img_np, ksize=33)

    final_image = vertical_projection(denoised_img)

    return final_image


def center_window(root, width, height):
    """使窗口居中显示"""
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_coordinate = (screen_width / 2) - (width / 2)
    y_coordinate = (screen_height / 2) - (height / 2)
    root.geometry(f"{width}x{height}+{int(x_coordinate)}+{int(y_coordinate)}")


def start_interface():
    """起始界面函数"""
    start_root = tk.Tk()
    start_root.title("手写数字识别系统")
    window_width = 500
    window_height = 500

    # 调用函数使窗口居中
    center_window(start_root, window_width, window_height)

    # 加载校徽图片
    try:
        school_logo = Image.open("school_logo.png")
        school_logo = school_logo.resize((150, 150))
        school_photo = ImageTk.PhotoImage(school_logo)
        logo_label = ttk.Label(start_root, image=school_photo)
        logo_label.image = school_photo  # 防止图片被垃圾回收
        logo_label.pack(pady=20)
    except FileNotFoundError:
        print("校徽图片未找到，请检查路径。")

    # 制作者信息和欢迎语
    welcome_font = ("华文新魏", 16)
    welcome_text = "欢迎使用手写数字识别系统\n\n制作者: 张峰榕"
    welcome_label = ttk.Label(start_root, text=welcome_text, justify=tk.CENTER, wraplength=300, font=welcome_font)
    welcome_label.pack(pady=20)

    def proceed_to_image():
        """关闭起始界面，打开图片识别界面"""
        start_root.destroy()  # 添加这一行来关闭起始界面
        img_interface()

    def proceed_to_handwriting():
        """关闭起始界面，打开手写识别界面"""
        start_root.destroy()  # 添加这一行来关闭起始界面
        handwriting_interface()

    def proceed_to_main():
        """关闭起始界面，打开成绩识别界面"""
        start_root.destroy()  # 添加这一行来关闭起始界面
        main_interface()

    def exit_app():
        """
        退出应用程序
        """
        start_root.destroy()

    # 创建进入图像识别界面按钮
    proceed_button = ttk.Button(start_root, text="图像识别", command=proceed_to_image)
    proceed_button.pack(ipadx=20, ipady=5, pady=5)

    # 创建进入手写识别界面按钮
    handwriting_button = ttk.Button(start_root, text="手写识别", command=proceed_to_handwriting)
    handwriting_button.pack(ipadx=20, ipady=5, pady=5)  # 调整位置

    # 创建进入成绩识别界面按钮
    handwriting_button = ttk.Button(start_root, text="成绩识别", command=proceed_to_main)
    handwriting_button.pack(ipadx=20, ipady=5, pady=5)

    # 创建退出按钮
    exit_button = ttk.Button(start_root, text="退出", command=exit_app)
    exit_button.pack(ipadx=5, ipady=5, pady=10)

    start_root.mainloop()


def img_interface():
    """图像识别界面函数"""

    def upload_image():
        """
        文件上传回调函数，处理用户上传的图片并显示识别结果及置信度
        """
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            image.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(image)
            img_label.config(image=photo)
            img_label.image = photo  # 防止图片被垃圾回收

            segmented_digits = image_process_image(file_path)
            predictions = predict_digits(segmented_digits)

            # 初始化识别结果字符串
            results_text = "识别结果:\n"

            # 遍历所有预测结果并构造显示文本
            for i, (predicted_digit, confidence) in enumerate(predictions, start=1):
                results_text += f"数字 {i}: {predicted_digit} (置信度: {confidence:.2f}%)\n"

            # 更新结果显示
            result_label.config(text=results_text)

    def clear_result():
        """
        清除预测结果、置信度
        """
        result_label.config(text="识别结果: ")
        img_label.config(image=None)  # 清除图片展示
        img_label.image = None  # 避免图片被垃圾回收

    def return_to_start():
        """
        关闭当前界面并返回起始界面
        """
        root.destroy()
        start_interface()

    # 创建主窗口并设置标题
    root = tk.Tk()
    root.title("手写数字识别系统 - 图像识别")
    window_width = 1000
    window_height = 700

    # 调用函数使窗口居中
    center_window(root, window_width, window_height)

    # 创建图片预览标签，增加边框
    img_label = ttk.Label(root, padding=10, relief=tk.RIDGE)
    img_label.pack(pady=20)

    # 创建结果展示标签
    result_label = ttk.Label(root, text="识别结果: ")
    result_label.pack(pady=(0, 10))

    # 创建上传按钮
    upload_button = ttk.Button(root, text="图片上传", command=upload_image)
    upload_button.pack(ipadx=5, ipady=5, pady=(0, 10))

    # 创建清除按钮
    clear_button = ttk.Button(root, text="清除结果", command=clear_result)
    clear_button.pack(ipadx=5, ipady=5, pady=(0, 10))  # 上下间距调整

    # 创建返回按钮
    return_button = ttk.Button(root, text="返回", command=return_to_start, style="TButton")
    return_button.pack(ipadx=5, ipady=5, pady=(0, 10))

    # 启动GUI主循环
    root.mainloop()


def handwriting_interface():
    """手写输入界面函数"""
    motion_id = None  # 确保在内部函数前声明了 motion_id
    draw_color = 'black'  # 初始化draw_color，默认颜色，可以从颜色选择中动态更改

    def start_drawing(event):
        """开始绘画事件处理"""
        nonlocal motion_id
        motion_id = canvas.bind('<B1-Motion>', draw)  # 绑定并存储ID
        x, y = event.x, event.y
        canvas.create_oval(x - 20, y - 20, x + 20, y + 20, fill=draw_color, outline=draw_color)

    def draw(event):
        """绘画事件处理"""
        x, y = event.x, event.y
        canvas.create_oval(x - 20, y - 20, x + 20, y + 20, fill=draw_color, outline=draw_color)

    def stop_drawing(event):
        """停止绘画事件处理"""
        nonlocal motion_id
        if motion_id:  # 检查ID是否存在，防止未绑定时尝试解绑
            canvas.unbind('<B1-Motion>', motion_id)
            motion_id = None

    def clear_canvas():
        """清空画布"""
        canvas.delete("all")

    def clear1_canvas():
        """
        清空画布
        清除预测结果、置信度
        """
        canvas.delete("all")
        result_label.config(text="识别结果: ")

    def save_canvas_as_image(canvas, filename):
        """保存画布内容为图像文件"""
        # 获取画布的边界尺寸
        x, y = canvas.winfo_rootx(), canvas.winfo_rooty()
        width, height = canvas.winfo_width(), canvas.winfo_height()

        # 截取整个画布区域
        img = ImageGrab.grab(bbox=(x + 140, y + 50, x + width + 250, y + height + 150))

        # 保存为PNG文件
        img.save(filename)

    def recognize_handwriting():
        """识别手写数字"""
        # 直接保存画布内容为图像
        save_canvas_as_image(canvas, "temp_handwriting.png")

        # 调用模型预测
        segmented_digits = handwriting_process_image("temp_handwriting.png")
        predictions = predict_digits(segmented_digits)

        # 初始化识别结果字符串
        results_text = "识别结果:\n"

        # 遍历所有预测结果并构造显示文本
        for i, (predicted_digit, confidence) in enumerate(predictions, start=1):
            results_text += f"数字 {i}: {predicted_digit} (置信度: {confidence:.2f}%)\n"

        # 更新结果显示
        result_label.config(text=results_text)

        # 删除临时文件
        os.remove("temp_handwriting.png")

    def set_color(new_color):
        """设置画笔颜色"""
        nonlocal draw_color
        draw_color = new_color

    def return_to_start():
        """
        关闭当前界面并返回起始界面
        """
        root.destroy()
        start_interface()

    # 创建主窗口
    root = tk.Tk()
    root.title("手写数字识别系统 - 手写识别")
    window_width = 1000
    window_height = 700

    # 调用函数使窗口居中
    center_window(root, window_width, window_height)

    # 画布设置
    canvas = tk.Canvas(root, width=500, height=500, bg='white')
    canvas.pack(pady=20)

    # 绑定画布事件
    canvas.bind('<Button-1>', start_drawing)

    # 颜色选择按钮
    colors = {'黑色': 'black', '红色': 'red', '黄色': 'yellow', '绿色': 'green', '蓝色': 'blue'}
    for color_name, color_code in colors.items():
        button = tk.Button(root, text=color_name, command=lambda c=color_code: set_color(c), bg=color_code, width=6,
                           height=1)
        button.pack(side=tk.LEFT, padx=5, pady=5)

    # 创建再写一次按钮
    clear_button = tk.Button(root, text="再写一次", command=clear_canvas)
    clear_button.pack(side=tk.LEFT, padx=5, pady=5)

    # 创建清空结果按钮
    clear1_button = tk.Button(root, text="清空结果", command=clear1_canvas)
    clear1_button.pack(side=tk.LEFT, padx=5, pady=5)

    # 创建识别按钮
    recognize_button = tk.Button(root, text="手写识别", command=recognize_handwriting)
    recognize_button.pack(side=tk.LEFT, padx=5, pady=5)

    # 创建结果标签
    result_label = tk.Label(root, text="识别结果: ")
    result_label.pack(pady=(0, 20))

    # 创建返回按钮
    return_button = ttk.Button(root, text="返回", command=return_to_start, style="TButton")
    return_button.pack(ipadx=5, ipady=5, pady=(0, 20))

    root.mainloop()


def main_interface():
    """成绩识别界面函数"""
    def upload_image():
        """
        文件上传回调函数，处理用户上传的图片并显示识别结果及置信度
        """
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            image.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(image)
            img_label.config(image=photo)
            img_label.image = photo  # 防止图片被垃圾回收

            segmented_digits = main_process_image(file_path)
            predictions = predict_digits(segmented_digits)

            # 初始化识别结果字符串
            results_text = "识别结果:\n"

            # 遍历所有预测结果并构造显示文本
            for i, (predicted_digit, confidence) in enumerate(predictions, start=1):
                results_text += f"成绩 {i}: {predicted_digit} (置信度: {confidence:.2f}%)\n"

            # 更新结果显示
            result_label.config(text=results_text)

    def clear_result():
        """
        清除预测结果、置信度
        """
        result_label.config(text="识别结果: ")
        img_label.config(image=None)  # 清除图片展示
        img_label.image = None  # 避免图片被垃圾回收

    def return_to_start():
        """
        关闭当前界面并返回起始界面
        """
        root.destroy()
        start_interface()

    # 创建主窗口并设置标题
    root = tk.Tk()
    root.title("手写数字识别系统 - 成绩识别")
    window_width = 1000
    window_height = 700

    # 调用函数使窗口居中
    center_window(root, window_width, window_height)

    # 创建图片预览标签，增加边框
    img_label = ttk.Label(root, padding=10, relief=tk.RIDGE)
    img_label.pack(pady=20)

    # 创建结果展示标签
    result_label = ttk.Label(root, text="识别结果: ")
    result_label.pack(pady=(0, 10))

    # 创建上传按钮
    upload_button = ttk.Button(root, text="成绩上传", command=upload_image)
    upload_button.pack(ipadx=5, ipady=5, pady=(0, 10))

    # 创建清除按钮
    clear_button = ttk.Button(root, text="清除结果", command=clear_result)
    clear_button.pack(ipadx=5, ipady=5, pady=(0, 10))  # 上下间距调整

    # 创建返回按钮
    return_button = ttk.Button(root, text="返回", command=return_to_start, style="TButton")
    return_button.pack(ipadx=5, ipady=5, pady=(0, 10))

    # 启动GUI主循环
    root.mainloop()

if __name__ == "__main__":
    start_interface()  # 首先显示起始界面
