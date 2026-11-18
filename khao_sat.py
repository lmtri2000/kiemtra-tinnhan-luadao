# Tổng hợp phiếu "PHIẾU THU THẬP Ý KIẾN VỀ TIN NHẮN RÁC" từ khao-sat/raw.py (câu trả lời chép từ Google Sheets)
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "khao-sat"))
from collections import Counter
from raw import ROWS, Q, T, N as NH, C, K, RN, TT, X, S, KB, OTT, SMS, CA
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"
n = len(ROWS); pc = lambda k: f"{k/n*100:.1f}%".replace(".", ",")
loai = Counter(x for r in ROWS for x in r[3]); kenh = Counter(r[4] for r in ROWS)
thang = Counter(r[2] for r in ROWS); nghe = Counter(r[1] for r in ROWS)
for ten, k in [("nghe rất nhiều", nghe[RN]), ("nghe thỉnh thoảng", nghe[TT]), ("tháng: thường xuyên", thang[X]),
               ("tháng: thỉnh thoảng", thang[S]), ("tháng: không", thang[KB]), ("quảng cáo/vay", loai[Q]),
               ("trúng thưởng", loai[T]), ("ngân hàng", loai[NH]), ("nhà nước", loai[C]), ("khác", loai[K]),
               ("ít nhất 1 dạng lừa đảo", sum(1 for r in ROWS if set(r[3]) & {T, NH, C})),
               ("chỉ OTT", kenh[OTT]), ("cả hai", kenh[CA]), ("chỉ SMS", kenh[SMS])]:
    print(f"{ten}: {k}/{n} = {pc(k)}")
ds1 = [("Quảng cáo, bất động sản,\nvay tiền nhanh", loai[Q]), ("Trúng thưởng, đòi phí\nnhận thưởng", loai[T]),
       ("Mạo danh ngân hàng\n(khóa tài khoản, OTP)", loai[NH]), ("Mạo danh cơ quan nhà nước\n(công an, thuế, VNeID)", loai[C]), ("Khác", loai[K])]
ds2 = [("Chỉ qua ứng dụng\n(Zalo, Messenger...)", kenh[OTT]), ("Cả SMS và ứng dụng", kenh[CA]), ("Chỉ qua SMS", kenh[SMS])]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
for a, ds, mau, tieu in [(ax[0], ds1, "#2b62a8", "Loại tin thường nhận (được chọn nhiều đáp án)"), (ax[1], ds2, "#d9822b", "Kênh nhận tin")]:
    ten = [t for t, _ in ds][::-1]; so = [s for _, s in ds][::-1]
    a.barh(ten, so, color=mau)
    for i, s in enumerate(so): a.text(s + 0.3, i, f"{s}/{n} ({pc(s)})", va="center", fontsize=10)
    a.set_xlim(0, n + 6); a.set_title(tieu, fontsize=12, fontweight="bold"); a.set_xlabel("Số người trả lời")
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "..", "anh", "khao_sat_thuc_trang.png"), dpi=200)
