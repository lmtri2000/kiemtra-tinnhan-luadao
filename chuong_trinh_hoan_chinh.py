# -*- coding: utf-8 -*-
"""
CHƯƠNG TRÌNH PHÁT HIỆN TIN NHẮN LỪA ĐẢO TIẾNG VIỆT (bản hoàn chỉnh của giáo trình)
Kết hợp: (1) bộ lọc theo luật (Bài 10) và (2) học máy Naive Bayes (Bài 11).
Cách chạy:
  - Trên Colab: tải file này và tin_nhan_mau.csv lên, rồi chạy ô:  %run chuong_trinh_hoan_chinh.py
  - Trên máy tính: cài Python, gõ  pip install pandas scikit-learn  rồi  python chuong_trinh_hoan_chinh.py
File tin_nhan_mau.csv phải nằm cùng thư mục.
"""

# ===== PHẦN 1. BỘ LỌC THEO LUẬT =====
import re

BANG_DIEM = {
    "trúng thưởng": 3, "trúng": 1, "nhận giải": 2, "quà": 1,
    "otp": 2, "mã xác thực": 2, "cvv": 3, "mật khẩu": 1, "số thẻ": 2,
    "chuyển khoản": 1, "gấp": 1, "khẩn": 1, "khóa": 2, "xác minh": 2,
    "phí": 2, "nạp": 2, "đặt cọc": 2,
    "việc nhẹ": 3, "lương cao": 2, "kiếm tiền": 2,
    "vay": 2, "giải ngân": 2, "hoàn tiền": 1, "miễn phí": 1, "lợi nhuận": 2,
}
TEN_MIEN_TIN_CAY = [".edu.vn", ".gov.vn"]

def tim_link_la(tin):
    link_la = []
    for link in re.findall(r"(?:https?://|www\.)\S+", tin.lower()):
        link = link.strip(".,!?")
        tin_cay = False
        for ten_mien in TEN_MIEN_TIN_CAY:
            if ten_mien in link:
                tin_cay = True
        if not tin_cay:
            link_la.append(link)
    return link_la

def giai_thich(tin):
    """Trả về (điểm, danh sách lý do)."""
    t = tin.lower()
    diem = 0
    ly_do = []
    for tu, d in BANG_DIEM.items():
        if tu in t:
            diem += d
            ly_do.append(f"có từ '{tu}' (+{d})")
    if len(tim_link_la(tin)) > 0:
        diem += 3
        ly_do.append("có đường link lạ (+3)")
    if re.search(r"\b0\d{9}\b", t):
        diem += 1
        ly_do.append("có số điện thoại (+1)")
    return diem, ly_do

def diem_luat(tin):
    return giai_thich(tin)[0]

def du_doan_luat(tin, nguong=1):
    if diem_luat(tin) >= nguong:
        return 1
    return 0


# ===== PHẦN 2. HỌC MÁY + PHẦN 3. CHƯƠNG TRÌNH HỎI ĐÁP =====
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# 1. Huấn luyện mô hình trên toàn bộ dữ liệu
df = pd.read_csv("tin_nhan_mau.csv")
mo_hinh_cuoi = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB())
mo_hinh_cuoi.fit(df["noi_dung"], df["nhan"])

LOI_KHUYEN = ("Không bấm vào đường link, không cung cấp mã OTP/mật khẩu, "
              "không chuyển tiền. Hãy gọi lại cho người thân hoặc tổng đài chính thức để kiểm tra.")

def kiem_tra_tin(tin):
    diem, ly_do = giai_thich(tin)
    luat_bao = du_doan_luat(tin, nguong=1) == 1
    p_lua = mo_hinh_cuoi.predict_proba([tin])[0][1]
    may_bao = p_lua >= 0.5
    if luat_bao and may_bao:
        muc = "🔴 NGUY HIỂM"
    elif luat_bao or may_bao:
        muc = "🟠 NGHI NGỜ"
    else:
        muc = "🟢 AN TOÀN"
    print("Kết luận:", muc)
    print(f"  Học máy: khả năng lừa đảo {p_lua:.0%}")
    print(f"  Bộ lọc luật: {diem} điểm nghi ngờ")
    for l in ly_do:
        print("    -", l)
    if muc != "🟢 AN TOÀN":
        print("  Lời khuyên:", LOI_KHUYEN)

# 2. Vòng lặp hỏi đáp
print("=== CHƯƠNG TRÌNH PHÁT HIỆN TIN NHẮN LỪA ĐẢO ===")
while True:
    tin = input("\nNhập tin nhắn (gõ 'thoát' để dừng): ")
    if tin.strip().lower() == "thoát":
        print("Cảm ơn bạn đã sử dụng chương trình!")
        break
    if tin.strip() == "":
        continue
    kiem_tra_tin(tin)
