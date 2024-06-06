import cv2 # thư viện OpenCV (xử lý ảnh)
import numpy as np 
import math as ma

# Thiết lập các tham số như kích cỡ cảu bộ lọc Gauss 
# Bộ lọc gauss ( làm mở và giảm nhiều 1  dạng của convolution (tích chập) giảm những tác nhân gây nhiễu gây mờ bằng cách dauwj vào các điểm ảnh đậm màu)
GAUSIAN_SMOOTH_FILTER_SIZE = (5,5) # ở đây ma trận chập sẽ là 5*5
# kích thước của ngưỡng thích nghi (Đây là kích thước của khu vực lân cận sẽ được sử dụng để tính ngưỡng cục bộ. Trong trường hợp này, kích thước của khu vực là 19x19 pixel.)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
# trọng số ngưỡng thích nghi (Đây là một hệ số được sử dụng trong phương pháp tính toán ngưỡng cục bộ. Nó ảnh hưởng đến mức độ làm mờ và làm sắc nét ảnh kết quả)
ADAPTIVE_THRESH_WEIGHT = 9


# hàm tiền xử lý ảnh gốc (
# mục tiêu là : trích xuất giá trị cường độ sáng, tối đa hóa tương phản và làm mịn ảnh)
def tienxuly(img):
    # trích xuất bàng hàm trích xuất cường độ sáng
    anh_xam = trich_xuat_gia_tri_cuong_do_sang(img)
    toidahoa = toi_da_hoa_anh(anh_xam) # ảnh màu xám
    # lưu vào file trong folder result  tên là anhsaucuonghoa.jpg
    cv2.imwrite("result/anhsaucuonghoa.png", toidahoa)
    dai, rong = anh_xam.shape
    anh_sau_khi_lam_min = cv2.GaussianBlur(toidahoa,GAUSIAN_SMOOTH_FILTER_SIZE,0)
    # tạo ảnh nhị phân
    anh_nhi_phan = cv2.adaptiveThreshold(anh_sau_khi_lam_min,255.0,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,ADAPTIVE_THRESH_BLOCK_SIZE,ADAPTIVE_THRESH_WEIGHT)
    return anh_xam, anh_nhi_phan

# hàm trích xuất cường độ sáng 
# mục đích là : 
# Hàm trích xuất giá trị cường độ sáng từ ảnh gốc bằng cách chuyển đổi từ không gian màu BGR sang HSV và tách lấy kênh giá trị cường độ sáng    
def trich_xuat_gia_tri_cuong_do_sang(img):
    cao, rong, numChannels = img.shape
    # chuyển đôi ảnh từ không gian màu (BGR) ==> HSV(vì giá trị này thường phản ánh độ sáng của mỗi pixel mà không bị ảnh hưởng bởi màu sắc)
    # Hệ màu HSV dùng để xử lý màu rất thuận tiện, vì mỗi màu sẽ có 1 giá trị Hue [0;360]
    # Hình ảnh đa số cấu tạo từ 3 kênh màu Red-Green-Blue
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # tách 3 tông màu của HSV ( hue , saturation, value)
    # Trong không gian màu HSV, giá trị (Value) đại diện cho cường độ sáng của một pixel, trong khi Hue đại diện cho màu sắc và Saturation đại diện cho độ bão hòa màu
    hue, saturation, value = cv2.split(imgHSV)
    return value #(cường độ sáng của một pixel)


# Hàm tối đa hóa độ tương phản của ảnh xám bằng cách áp dụng các phép toán hình thái học Top-Hat và Black-Hat.
# mục đích là làm vùng tối thì tối hơn vung trắng thì càn trắng hơn
def toi_da_hoa_anh(img):
    cao, rong = img.shape
    # Tạo bộ lọc hình chữ nhật
    # được sử dụng để tạo ra các phần tử cấu trúc (structuring elements) cho các phép biến đổi hình thái học,
    # chẳng hạn như 
    # giãn nở (dilation) cho đối tượng ban đầu trong ảnh tăng lên về kích thước 
    # co lại (erosion) giảm kích thước của đối tượng, tách rời các đối tượng gần nhau, làm mảnh và tìm xương của đối tượng
    # mở (opening) xóa các điểm ảnh nhiều xung quanh hình
    # đóng (closing). xóa các điểm nhiễu bên trong hình

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # tophat (phép trừ ảnh của ảnh ban đầu với ảnh sau khi thực hiện phép mở) ==> nổi bật nhưng chi tiết trắng trong nền tối
    img_top_hat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, structuringElement, iterations=10) # số lần lặp lại cho phép biến đổi hình thái học iterations=10
    cv2.imwrite("result/anhsautophat.png", img_top_hat)
    
    # backhat (Nổi bật chi tiết tối trong nền sáng)
    img_back_hat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT, structuringElement, iterations=10)
    cv2.imwrite("result/anhsaubackhat.png", img_back_hat)
    # áp dụng công thức + top - back:
    img_sau_chinh = cv2.add(img,img_top_hat)
    img_sau_chinh = cv2.subtract(img_sau_chinh,img_back_hat)

    # hiện ảnh sau chỉnh
    
    return img_sau_chinh


