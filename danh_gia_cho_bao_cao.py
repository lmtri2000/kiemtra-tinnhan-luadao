# -*- coding: utf-8 -*-
"""
ĐÁNH GIÁ ĐẦY ĐỦ CHO BUỔI BÁO CÁO KHKT
Đề tài: Xây dựng hệ thống nhận diện và cảnh báo tin nhắn rác, tin nhắn lừa đảo bằng tiếng Việt sử dụng thuật toán Naive Bayes trên nền tảng Python.

Mọi con số trong bài thuyết trình và trong slide đều được in ra từ file này.
Cách chạy (Colab hoặc máy tính):  python danh_gia_cho_bao_cao.py
Cần file tin_nhan_mau.csv nằm cùng thư mục.  Thư viện: pandas, scikit-learn, matplotlib.

Kết quả:
  - in ra màn hình (và ghi vào ket_qua.txt)
  - ket_qua.json  : các con số, để chèn vào slide
  - mo_hinh_web.json : tham số mô hình đã huấn luyện, dùng cho trang demo
  - các hình .png trong thư mục anh/
"""
import json, os, re, sys, unicodedata
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score)

THU_MUC = os.path.dirname(os.path.abspath(__file__))
os.chdir(THU_MUC)
os.makedirs("anh", exist_ok=True)
log = open("ket_qua.txt", "w", encoding="utf-8")

def in_ra(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.write(s + "\n")

# ===================== BỘ LỌC THEO LUẬT (giống hệt chương trình hoàn chỉnh) =====================
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
        if not any(tm in link for tm in TEN_MIEN_TIN_CAY):
            link_la.append(link)
    return link_la

def giai_thich(tin):
    t = tin.lower()
    diem, ly_do = 0, []
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
    return 1 if diem_luat(tin) >= nguong else 0

def chon_nguong(X_hoc, y_hoc):
    """Chọn ngưỡng 1..6 cho tỉ lệ đúng cao nhất, CHỈ trên tập học."""
    tot, ti_le_tot = 1, -1
    for ng in range(1, 7):
        tl = np.mean([du_doan_luat(t, ng) == n for t, n in zip(X_hoc, y_hoc)])
        if tl > ti_le_tot:
            tot, ti_le_tot = ng, tl
    return tot

def tao_mo_hinh():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB())

def bo_dau(s):
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def chi_so(y, d):
    tn, fp, fn, tp = confusion_matrix(y, d, labels=[0, 1]).ravel()
    return dict(acc=accuracy_score(y, d), prec=precision_score(y, d, zero_division=0),
                rec=recall_score(y, d, zero_division=0), f1=f1_score(y, d, zero_division=0),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

def dong(ten, c):
    in_ra(f"{ten:22s} | {c['acc']:6.1%} | {c['prec']:6.1%} | {c['rec']:6.1%} | {c['f1']:6.1%} |"
          f"  {c['fp']:2d}  |  {c['fn']:2d}")

TIEU_DE = "Phương pháp            | Accur. | Precis.| Recall |   F1   |  FP  |  FN"
KQ = {}

# ===================== 1. DỮ LIỆU =====================
df = pd.read_csv("tin_nhan_mau.csv")
df["do_dai"] = df["noi_dung"].str.split().str.len()
in_ra("=" * 72)
in_ra("1. DỮ LIỆU")
in_ra(f"Tổng số tin: {len(df)} | lừa đảo: {(df.nhan == 1).sum()} | bình thường: {(df.nhan == 0).sum()}")
dd = df.groupby("nhan")["do_dai"].mean()
in_ra(f"Độ dài trung bình (số âm tiết): bình thường {dd[0]:.1f}, lừa đảo {dd[1]:.1f}")
co_link = df["noi_dung"].apply(lambda t: len(tim_link_la(t)) > 0)
co_sdt = df["noi_dung"].str.lower().str.contains(r"\b0\d{9}\b", regex=True)
for ten, cot in [("có link lạ", co_link), ("có số điện thoại", co_sdt)]:
    in_ra(f"Tỉ lệ tin {ten}: lừa đảo {cot[df.nhan == 1].mean():.1%}, bình thường {cot[df.nhan == 0].mean():.1%}")
KQ["du_lieu"] = dict(tong=len(df), lua=int((df.nhan == 1).sum()), thuong=int((df.nhan == 0).sum()),
                    do_dai_thuong=round(dd[0], 1), do_dai_lua=round(dd[1], 1),
                    link_lua=float(co_link[df.nhan == 1].mean()), link_thuong=float(co_link[df.nhan == 0].mean()),
                    sdt_lua=float(co_sdt[df.nhan == 1].mean()), sdt_thuong=float(co_sdt[df.nhan == 0].mean()))

# ===================== 2. CHIA 70/30 =====================
in_ra("=" * 72)
in_ra("2. THÍ NGHIỆM CHÍNH: chia 70% học / 30% kiểm tra (random_state=42, stratify)")
X_hoc, X_kt, y_hoc, y_kt = train_test_split(df["noi_dung"], df["nhan"], test_size=0.3,
                                            random_state=42, stratify=df["nhan"])
X_hoc, X_kt, y_hoc, y_kt = list(X_hoc), list(X_kt), list(y_hoc), list(y_kt)
in_ra(f"Tập học: {len(X_hoc)} tin, tập kiểm tra: {len(X_kt)} tin ({sum(y_kt)} lừa đảo)")
ng = chon_nguong(X_hoc, y_hoc)
in_ra("Ngưỡng bộ luật chọn trên tập học:", ng)
d_luat = [du_doan_luat(t, ng) for t in X_kt]
mh = tao_mo_hinh().fit(X_hoc, y_hoc)
p_kt = mh.predict_proba(X_kt)[:, 1]
d_nb = list((p_kt >= 0.5).astype(int))
d_hoac = [max(a, b) for a, b in zip(d_luat, d_nb)]
d_va = [min(a, b) for a, b in zip(d_luat, d_nb)]
in_ra(TIEU_DE)
KQ["chia_70_30"] = {}
for ten, khoa, d in [("Bộ luật", "luat", d_luat), ("Naive Bayes", "nb", d_nb),
                     ("Kết hợp HOẶC", "hoac", d_hoac), ("Kết hợp VÀ", "va", d_va)]:
    c = chi_so(y_kt, d); dong(ten, c); KQ["chia_70_30"][khoa] = c
KQ["nguong_luat"] = ng

in_ra("\nCác tin bị đoán sai trên tập kiểm tra:")
sai = []
for tin, y, a, b in zip(X_kt, y_kt, d_luat, d_nb):
    if a != y or b != y:
        nhan = []
        if a != y: nhan.append("Luật " + ("báo nhầm" if a == 1 else "bỏ lọt"))
        if b != y: nhan.append("NB " + ("báo nhầm" if b == 1 else "bỏ lọt"))
        in_ra(f"  [{'lừa đảo' if y else 'bình thường'}] [{', '.join(nhan)}] {tin}")
        sai.append(dict(tin=tin, that=int(y), luat=int(a), nb=int(b)))
KQ["tin_sai"] = sai

in_ra("\nNB với các ngưỡng xác suất khác nhau:")
in_ra(TIEU_DE)
KQ["nguong_nb"] = {}
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    c = chi_so(y_kt, list((p_kt >= t).astype(int))); dong(f"NB ngưỡng {t}", c); KQ["nguong_nb"][str(t)] = c

# ===================== 3. KIỂM ĐỊNH CHÉO 5 LẦN =====================
in_ra("=" * 72)
in_ra("3. KIỂM ĐỊNH CHÉO 5 LẦN (StratifiedKFold, shuffle, random_state=42)")
in_ra("   Mỗi lần: ngưỡng luật chọn trên 4 phần học; NB học trên 4 phần; chấm trên phần còn lại.")
chia5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_all, y_all = df["noi_dung"].values, df["nhan"].values
cv = {k: [] for k in ["luat", "nb", "hoac"]}
for i, (ih, ik) in enumerate(chia5.split(X_all, y_all), 1):
    ng_i = chon_nguong(X_all[ih], y_all[ih])
    a = [du_doan_luat(t, ng_i) for t in X_all[ik]]
    b = list(tao_mo_hinh().fit(X_all[ih], y_all[ih]).predict(X_all[ik]))
    h = [max(u, v) for u, v in zip(a, b)]
    for k, d in [("luat", a), ("nb", b), ("hoac", h)]:
        cv[k].append(chi_so(y_all[ik], d))
    in_ra(f"  Lần {i}: ngưỡng luật = {ng_i} | Acc luật {cv['luat'][-1]['acc']:.1%}, "
          f"NB {cv['nb'][-1]['acc']:.1%}, HOẶC {cv['hoac'][-1]['acc']:.1%}")
KQ["cv"] = {}
in_ra("Trung bình ± độ lệch chuẩn qua 5 lần:")
for k, ten in [("luat", "Bộ luật"), ("nb", "Naive Bayes"), ("hoac", "Kết hợp HOẶC")]:
    r = {m: (float(np.mean([c[m] for c in cv[k]])), float(np.std([c[m] for c in cv[k]])))
         for m in ["acc", "prec", "rec", "f1"]}
    r["fp"] = int(sum(c["fp"] for c in cv[k])); r["fn"] = int(sum(c["fn"] for c in cv[k]))
    r["acc_tung_lan"] = [c["acc"] for c in cv[k]]
    KQ["cv"][k] = r
    in_ra(f"  {ten:14s}: Acc {r['acc'][0]:.1%} ± {r['acc'][1]:.1%} | Prec {r['prec'][0]:.1%} | "
          f"Recall {r['rec'][0]:.1%} ± {r['rec'][1]:.1%} | F1 {r['f1'][0]:.1%} | "
          f"tổng báo nhầm {r['fp']}/70, tổng bỏ lọt {r['fn']}/70")

# ===================== 4. KẺ GIAN VIẾT KHÔNG DẤU =====================
in_ra("=" * 72)
in_ra("4. THÍ NGHIỆM 'KẺ GIAN VIẾT KHÔNG DẤU' (bỏ dấu 42 tin kiểm tra, mô hình giữ nguyên)")
X_kt_kd = [bo_dau(t) for t in X_kt]
d_luat_kd = [du_doan_luat(t, ng) for t in X_kt_kd]
d_nb_kd = list(mh.predict(X_kt_kd))
d_hoac_kd = [max(a, b) for a, b in zip(d_luat_kd, d_nb_kd)]
in_ra(TIEU_DE)
KQ["khong_dau"] = {}
for ten, khoa, d in [("Bộ luật (không dấu)", "luat", d_luat_kd), ("NB (không dấu)", "nb", d_nb_kd),
                     ("HOẶC (không dấu)", "hoac", d_hoac_kd)]:
    c = chi_so(y_kt, d); dong(ten, c); KQ["khong_dau"][khoa] = c
# Cách khắc phục: bỏ dấu cả lúc học và lúc kiểm tra
mh_kd = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), preprocessor=lambda s: bo_dau(s.lower())),
                      MultinomialNB()).fit(X_hoc, y_hoc)
c = chi_so(y_kt, list(mh_kd.predict(X_kt_kd)))
dong("NB học không dấu", c); KQ["khong_dau"]["nb_sua"] = c
c2 = chi_so(y_kt, list(mh_kd.predict(X_kt)))
dong("NB học không dấu, KT có dấu", c2); KQ["khong_dau"]["nb_sua_codau"] = c2

# ===================== 5. MÔ HÌNH CUỐI + TỪ ĐẶC TRƯNG =====================
in_ra("=" * 72)
in_ra("5. MÔ HÌNH CUỐI (học trên cả 140 tin) VÀ CÁC CỤM TỪ MÁY COI LÀ 'ĐÁNG NGỜ' NHẤT")
mo_hinh_cuoi = tao_mo_hinh().fit(df["noi_dung"], df["nhan"])
vec, nb = mo_hinh_cuoi.named_steps["tfidfvectorizer"], mo_hinh_cuoi.named_steps["multinomialnb"]
ten_dt = vec.get_feature_names_out()
chenh = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
thu_tu = np.argsort(chenh)
top_lua = [(ten_dt[i], float(chenh[i])) for i in thu_tu[::-1][:12]]
top_thuong = [(ten_dt[i], float(chenh[i])) for i in thu_tu[:12]]
in_ra(f"Số đặc trưng (âm tiết + cặp âm tiết): {len(ten_dt)}")
in_ra("Nghiêng về LỪA ĐẢO :", ", ".join(f"{w} ({v:+.2f})" for w, v in top_lua))
in_ra("Nghiêng về BÌNH THƯỜNG:", ", ".join(f"{w} ({v:+.2f})" for w, v in top_thuong))
KQ["so_dac_trung"] = len(ten_dt)
KQ["top_lua"], KQ["top_thuong"] = top_lua, top_thuong

LOI_KHUYEN = ("Không bấm vào đường link, không cung cấp mã OTP/mật khẩu, không chuyển tiền. "
              "Hãy gọi lại cho người thân hoặc tổng đài chính thức để kiểm tra.")
def kiem_tra_tin(tin):
    diem, ly_do = giai_thich(tin)
    luat_bao = du_doan_luat(tin, 1) == 1
    p = float(mo_hinh_cuoi.predict_proba([tin])[0][1])
    may_bao = p >= 0.5
    muc = "NGUY HIỂM" if (luat_bao and may_bao) else ("NGHI NGỜ" if (luat_bao or may_bao) else "AN TOÀN")
    return dict(tin=tin, muc=muc, p=p, diem=diem, ly_do=ly_do)

# Tin demo: KHÔNG có trong 140 tin dữ liệu (đã kiểm tra ở dưới)
TIN_DEMO = [
    "Chúc mừng bạn trúng thưởng 50 triệu, bấm http://nhanqua.xyz và gửi mã OTP để nhận giải",
    "Thứ 7 này lớp mình đi dã ngoại nhé, nhớ mang nón",
    "Con đang cấp cứu ở bệnh viện, mẹ chuyển khoản gấp 10 triệu vào số tài khoản này",
    "Tài khoản ngân hàng của quý khách bị khóa, truy cập www.vietcom-bank.top để xác minh",
    "Tuyển cộng tác viên việc nhẹ lương cao, làm tại nhà, nhận 500k mỗi ngày",
    "Mẹ ơi tối nay con về muộn, mẹ đừng chờ cơm nhé",
    "Lịch thi học kỳ I đã đăng trên trang thpt.edu.vn, các em xem kỹ",
    "Chuc mung ban trung thuong, gui ma OTP de nhan qua",
    "Nhà trường thông báo học phí học kỳ 2 nộp tại phòng kế toán trước ngày 15",
    "Ngân hàng thông báo số dư tài khoản của quý khách vừa tăng 2.000.000đ",
]
in_ra("\nThử chương trình với các tin demo (không có trong dữ liệu học):")
tap = set(df["noi_dung"])
KQ["demo"] = []
for t in TIN_DEMO:
    assert t not in tap
    r = kiem_tra_tin(t); KQ["demo"].append(r)
    in_ra(f"  [{r['muc']:9s}] NB {r['p']:.0%} | luật {r['diem']} điểm | {t}")

# Xuất tham số cho trang demo web (trang web tính lại đúng như sklearn)
vocab = {k: int(v) for k, v in vec.vocabulary_.items()}
json.dump(dict(vocab=vocab, idf=[float(x) for x in vec.idf_],
               flp=[[float(x) for x in r] for r in nb.feature_log_prob_],
               clp=[float(x) for x in nb.class_log_prior_],
               bang_diem=BANG_DIEM, ten_mien_tin_cay=TEN_MIEN_TIN_CAY,
               kiem_tra={t: float(mo_hinh_cuoi.predict_proba([t])[0][1])
                         for t in list(df["noi_dung"]) + TIN_DEMO}),
          open("mo_hinh_web.json", "w", encoding="utf-8"), ensure_ascii=False)

json.dump(KQ, open("ket_qua.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ===================== 6. HÌNH VẼ =====================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"
    # 6a. Ma trận nhầm lẫn 3 phương pháp
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, (ten, k) in zip(ax, [("Bộ luật", "luat"), ("Naive Bayes", "nb"), ("Kết hợp HOẶC", "hoac")]):
        c = KQ["chia_70_30"][k]
        m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        a.imshow(m, cmap="Blues", vmin=0, vmax=21)
        for i in range(2):
            for j in range(2):
                a.text(j, i, m[i, j], ha="center", va="center", fontsize=22,
                       color="white" if m[i, j] > 10 else "black", fontweight="bold")
        a.set_xticks([0, 1], ["Bình thường", "Lừa đảo"]); a.set_yticks([0, 1], ["Bình thường", "Lừa đảo"])
        a.set_xlabel("Máy đoán"); a.set_ylabel("Thật ra"); a.set_title(ten, fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig("anh/ma_tran_nham_lan_3.png", dpi=200); plt.close()
    # 6b. Kiểm định chéo
    fs = lambda v: f"{v:.1f}".replace(".", ",")
    fig, a = plt.subplots(figsize=(9, 4.6))
    mau = {"luat": "#7a8a99", "nb": "#2b6cb0", "hoac": "#c0392b"}
    ten = {"luat": "Bộ luật", "nb": "Naive Bayes", "hoac": "Kết hợp HOẶC"}
    x = np.arange(4); w = 0.26
    for i, k in enumerate(["luat", "nb", "hoac"]):
        v = [KQ["cv"][k][m][0] * 100 for m in ["acc", "prec", "rec", "f1"]]
        a.bar(x + (i - 1) * w, v, w, color=mau[k], label=ten[k])
        for xx, vv in zip(x + (i - 1) * w, v):
            a.text(xx, vv + 1.5, fs(vv), ha="center", fontsize=9)
    a.set_xticks(x, ["Accuracy", "Precision", "Recall", "F1"]); a.set_ylim(0, 112); a.set_ylabel("%")
    a.set_title("Kiểm định chéo 5 lần (trung bình)", fontweight="bold", pad=28)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=3, fontsize=10, frameon=False)
    a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig("anh/kiem_dinh_cheo.png", dpi=200); plt.close()
    # 6c. Không dấu
    fig, a = plt.subplots(figsize=(8, 4.4))
    nh = ["Bộ luật", "Naive Bayes", "Kết hợp HOẶC"]
    co = [KQ["chia_70_30"][k]["rec"] * 100 for k in ["luat", "nb", "hoac"]]
    kd = [KQ["khong_dau"][k]["rec"] * 100 for k in ["luat", "nb", "hoac"]]
    x = np.arange(3)
    a.bar(x - 0.18, co, 0.36, color="#2b6cb0", label="Tin có dấu")
    a.bar(x + 0.18, kd, 0.36, color="#e67e22", label="Cùng tin, bỏ dấu")
    for xx, v in list(zip(x - 0.18, co)) + list(zip(x + 0.18, kd)):
        a.text(xx, v + 1.5, fs(v) + "%", ha="center", fontsize=10, fontweight="bold")
    a.set_xticks(x, nh); a.set_ylim(0, 115); a.set_ylabel("Recall (% tin lừa đảo bắt được)")
    a.set_title("Kẻ gian viết không dấu", fontweight="bold", pad=28)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=2, frameon=False)
    a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig("anh/khong_dau.png", dpi=200); plt.close()
    in_ra("\nĐã lưu hình vào thư mục anh/")
except ImportError:
    in_ra("(Không có matplotlib, bỏ qua phần vẽ hình)")
log.close()
