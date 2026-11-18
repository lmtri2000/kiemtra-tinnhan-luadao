# -*- coding: utf-8 -*-
# =====================================================================
#  MÁY SOI TIN NHẮN LỪA ĐẢO: CHUYÊN GIA 1 (bộ luật) + CHUYÊN GIA 2 (Naive Bayes)
#  Bản tách từng phần để học. Cho đúng các con số của trang demo.
#  Chạy:  python hai_chuyen_gia.py      (cần tin_nhan_mau.csv cùng thư mục)
# =====================================================================

# ---------- PHẦN 0. NẠP THƯ VIỆN ----------
import re, math
import pandas as pd
from collections import Counter

# ---------- PHẦN 1. ĐỌC DỮ LIỆU ----------
df = pd.read_csv("tin_nhan_mau.csv")          # 2 cột: noi_dung, nhan (1 = lừa đảo, 0 = bình thường)
cac_tin = list(df["noi_dung"])
cac_nhan = list(df["nhan"])
print("Số tin:", len(cac_tin), "| lừa đảo:", sum(cac_nhan), "| bình thường:", len(cac_nhan) - sum(cac_nhan))

# ---------- PHẦN 2. CHUYÊN GIA 1: BỘ LUẬT CHẤM ĐIỂM ----------
BANG_DIEM = {
    "trúng thưởng": 3, "trúng": 1, "nhận giải": 2, "quà": 1,
    "otp": 2, "mã xác thực": 2, "cvv": 3, "mật khẩu": 1, "số thẻ": 2,
    "chuyển khoản": 1, "gấp": 1, "khẩn": 1, "khóa": 2, "xác minh": 2,
    "phí": 2, "nạp": 2, "đặt cọc": 2,
    "việc nhẹ": 3, "lương cao": 2, "kiếm tiền": 2,
    "vay": 2, "giải ngân": 2, "hoàn tiền": 1,
    "miễn phí": 1, "lợi nhuận": 2,
}
TEN_MIEN_TIN_CAY = [".edu.vn", ".gov.vn"]

def tim_link_la(tin):
    link_la = []
    for link in re.findall(r"(?:https?://|www\.)\S+", tin.lower()):
        link = link.strip(".,!?")
        if not any(tm in link for tm in TEN_MIEN_TIN_CAY):
            link_la.append(link)
    return link_la

def chuyen_gia_1(tin):
    t = tin.lower()
    diem, ly_do = 0, []
    for tu, d in BANG_DIEM.items():
        if tu in t:
            diem += d
            ly_do.append(f"'{tu}' +{d}")
    if len(tim_link_la(tin)) > 0:
        diem += 3
        ly_do.append("link lạ +3")
    if re.search(r"\b0\d{9}\b", t):
        diem += 1
        ly_do.append("số điện thoại +1")
    return diem, ly_do

# ---------- PHẦN 3. CHUYÊN GIA 2, BƯỚC A: TÁCH TIN THÀNH "TỪ" ----------
def tach_tu(tin):
    am_tiet = re.findall(r"\b\w\w+\b", tin.lower())               # âm tiết có từ 2 ký tự
    cap = [am_tiet[i] + " " + am_tiet[i + 1]
           for i in range(len(am_tiet) - 1)]   # cặp âm tiết liền nhau
    return am_tiet + cap

# ---------- PHẦN 4. CHUYÊN GIA 2, BƯỚC B: TF-IDF (TỪ NÀO QUAN TRỌNG) ----------
N = len(cac_tin)
so_tin_chua = Counter()                      # df(w): bao nhiêu tin có chứa từ w
for tin in cac_tin:
    so_tin_chua.update(set(tach_tu(tin)))
TU_DIEN = sorted(so_tin_chua)                # danh sách mọi "từ" máy biết
IDF = {w: math.log((1 + N) / (1 + so_tin_chua[w])) + 1 for w in TU_DIEN}

def tf_idf(tin):
    dem = Counter(w for w in tach_tu(tin) if w in IDF)     # tf: số lần xuất hiện (bỏ từ lạ)
    x = {w: dem[w] * IDF[w] for w in dem}                    # tf × idf
    do_dai = math.sqrt(sum(v * v for v in x.values()))      # độ dài vectơ
    return {w: v / do_dai for w, v in x.items()} if do_dai > 0 else {}

# ---------- PHẦN 5. CHUYÊN GIA 2, BƯỚC C: HỌC (ĐẾM THEO TỪNG LOẠI) ----------
tong_trong_so = {0: Counter(), 1: Counter()}
for tin, nhan in zip(cac_tin, cac_nhan):
    tong_trong_so[nhan].update(tf_idf(tin))
V = len(TU_DIEN)
P_LOAI = {c: cac_nhan.count(c) / N for c in (0, 1)}            # P(lừa đảo), P(bình thường)
TONG = {c: sum(tong_trong_so[c].values()) for c in (0, 1)}

def p_tu(w, c):                                                # P(w | loại c), làm trơn Laplace (+1)
    return (tong_trong_so[c][w] + 1) / (TONG[c] + V)

# ---------- PHẦN 6. CHUYÊN GIA 2, BƯỚC D: CÔNG THỨC BAYES ----------
def chuyen_gia_2(tin):
    x = tf_idf(tin)
    diem = {}
    for c in (0, 1):
        diem[c] = math.log(P_LOAI[c]) + sum(
            v * math.log(p_tu(w, c)) for w, v in x.items())
    p_lua_dao = 1 / (1 + math.exp(diem[0] - diem[1]))
    return p_lua_dao, x, diem

# ---------- PHẦN 7. KẾT HỢP HAI CHUYÊN GIA ----------
def ket_luan(tin):
    diem, ly_do = chuyen_gia_1(tin)
    p, _, _ = chuyen_gia_2(tin)
    cg1, cg2 = diem >= 1, p >= 0.5
    if cg1 and cg2:
        muc = "NGUY HIỂM"
    elif cg1 or cg2:
        muc = "NGHI NGỜ"
    else:
        muc = "AN TOÀN"
    return muc, diem, ly_do, p

# ---------- PHẦN 8. CHẠY THỬ ----------
if __name__ == "__main__":
    for tin in ["Mẹ ơi chuyển khoản gấp cho con",
                "Con đang cấp cứu ở bệnh viện, mẹ chuyển khoản gấp 10 triệu vào số tài khoản này",
                "Bạn trúng thưởng, nộp phí nhận quà ngay",
                "Chiều nay con về muộn, mẹ đừng chờ cơm"]:
        muc, diem, ly_do, p = ket_luan(tin)
        print(f"\n{tin}\n  Chuyên gia 1: {diem} điểm {ly_do}\n  Chuyên gia 2: {p:.1%}\n  => {muc}")
