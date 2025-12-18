Dự án xây dựng mô hình lai LSTM-CNN nhằm dự đoán giá trị chuỗi thời gian dựa trên việc kết hợp khả năng trích xuất đặc trưng không gian của mạng CNN và khả năng học phụ thuộc thời gian dài hạn của mạng LSTM
Từ dữ liệu thô ban đầu, hệ thống thực hiện tính toán và trích xuất 11 chỉ số kỹ thuật đặc trưng, sau đó tổ chức dữ liệu dưới dạng cửa sổ trượt với độ dài 90 bước thời gian cho mỗi mẫu đầu vào. 
Quy trình này giúp mô hình nắm bắt được các biến động phức tạp và xu hướng tiềm ẩn trong dữ liệu lịch sử, từ đó tối ưu hóa độ chính xác cho kết quả dự đoán giá cuối cùng
