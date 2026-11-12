# SED-L zero-shot trên R2S100K train

Project này gọi checkpoint **SED ConvNeXt-L chính thức** để xuất mask 15 ID R2S100K, rồi đánh giá trên tập train có nhãn. Không cập nhật trọng số. Kết quả là zero-shot transfer trên **train**, không phải test độc lập. Mã SED gốc là Git submodule ở `third_party/SED`, commit `db20d93c0ec8b0ce7fae267a10f826dfc7da15f5` ([nguồn](https://github.com/xb534/SED)). Khi clone project qua Git, dùng `git clone --recurse-submodules` hoặc chạy `git submodule update --init --recursive`. Taxonomy, palette và prompt nằm trong `sed_r2s100k/taxonomy.py`; prompt ensemble `imagenet` theo cấu hình SED-L.

## Cài đặt

Trên máy chạy RTX 3090, cần Python 3.10, CUDA driver tương thích CUDA 11.8, trình biên dịch C++ và git:

```bash
bash scripts/setup_env.sh
```

Script tạo `.venv` riêng, cài PyTorch 2.1.2, torchvision 0.16.2, Pillow 9.5.0, Detectron2 v0.6, TensorBoard 2.14.1 và bản OpenCLIP đi kèm SED. Mã nguồn SED được khóa ở commit trên. Khi khởi tạo, pipeline gom ImageNet text prompts thành batch và mã hóa 14 lớp vùng đường cùng 8 khái niệm cảnh background; trọng số model được giữ nguyên. Có thể đặt `MAX_JOBS=2` nếu RAM host hạn chế.

Tải **checkpoint SED (L)** từ [README chính thức](https://github.com/xb534/SED#results) (liên kết Google Drive dòng SED (L)) vào `checkpoints/sed_model_large.pth`. Đây phải là checkpoint model hoàn chỉnh; bước nạp sẽ kiểm tra encoder và decoder. Lần đầu chạy, OpenCLIP có thể tải thêm trọng số khởi tạo qua mạng. Đặt `dataset_root`, `checkpoint`, `output`, `device`, `workers` trong `config.yaml` cho server.

## Chuyển từ local lên server

Trên local, từ thư mục project, thay `minhduc@server.example.com` và `/home/minhduc/work` bằng tài khoản SSH và thư mục thật trên server:

```bash
bash scripts/sync_to_server.sh minhduc@server.example.com /home/minhduc/work
```

Script đồng bộ mã nguồn (gồm SED gốc), checkpoint, `train/` và `train_labels/` qua `rsync`, hỗ trợ chạy lại khi đường truyền gián đoạn. Nó bỏ `.venv` và các kết quả `runs/` ở local. Nếu dữ liệu nằm nơi khác, truyền đường dẫn local làm tham số thứ ba. Đặt `SSH_PORT=2222` nếu SSH không dùng cổng 22; `RSYNC_DRY_RUN=1` để xem trước danh sách chuyển (bước này vẫn tạo thư mục đích trên server). Script kiểm tra SHA256 checkpoint sau khi đồng bộ. Không cần chuyển các split val/test cho lượt đánh giá train này.

Trên server:

```bash
cd /home/minhduc/work/SED
bash scripts/setup_server.sh
```

Script kiểm tra GPU, cài `.venv`, xác nhận dependencies và commit SED, tạo `config.server.yaml` với đường dẫn server, sau đó infer và evaluate một ảnh thật. Kết quả smoke ở `runs/server_smoke_*/`. Máy chủ cần Python 3.10, C++ compiler, git, driver NVIDIA, quyền tải package từ Internet và khoảng 8 GB chỗ trống cho project, checkpoint, dữ liệu (chưa tính `.venv` và kết quả). Có thể đặt `DATASET_ROOT`, `CHECKPOINT`, `OUTPUT`, `SMOKE_IMAGE`, `WORKERS` qua biến môi trường nếu dùng bố cục khác. Sau khi smoke thành công, chạy:

```bash
.venv/bin/python -m sed_r2s100k --config config.server.yaml benchmark
.venv/bin/python -m sed_r2s100k --config config.server.yaml run
```

Chạy `run` khi đã xem báo cáo benchmark; pipeline vẫn tự dừng nếu dự báo vượt 3 giờ.

## Chạy

```bash
.venv/bin/python -m sed_r2s100k prepare
.venv/bin/python -m sed_r2s100k benchmark
.venv/bin/python -m sed_r2s100k run
```

`run` tự chuẩn bị manifest, warm-up 10 ảnh, đo 100 ảnh rải đều, rồi dự báo thời gian inference + đánh giá với biên 20%. Nếu dự báo vượt 3 giờ, nó dừng ở `benchmark.json`. Khi đáp ứng, nó chạy toàn bộ 9.562 ảnh train. Để kiểm tra đúng một ảnh mà không quét toàn bộ tập train, truyền đường dẫn ảnh vào `--image`:

```bash
.venv/bin/python -m sed_r2s100k infer --image '/run/media/minhduc/D/Study/r2s100k/train/<ten-anh>.jpg' --output runs/local_smoke --workers 0
```

Lệnh này ghi manifest chỉ chứa đúng ảnh đã chọn và không quét 9.562 ảnh. Sau inference, đánh giá chính ảnh đó bằng lệnh: `.venv/bin/python -m sed_r2s100k evaluate --image '/run/media/minhduc/D/Study/r2s100k/train/<ten-anh>.jpg' --output runs/local_smoke`. Có thể dùng `infer` và `evaluate` riêng nếu muốn kiểm soát từng bước:

```bash
.venv/bin/python -m sed_r2s100k infer
.venv/bin/python -m sed_r2s100k evaluate
```

`infer --image` tạo manifest đơn ảnh ngay lập tức mà không đọc ground truth; `evaluate --image` ghép đúng mask tương ứng. Nếu không truyền `--image`, inference không manifest vẫn tạo manifest đầy đủ cho cả split. `evaluate` tự chuẩn bị manifest nhãn khi thiếu. Khi smoke test, dùng `.venv/bin/python -m sed_r2s100k infer --output runs/local_smoke --max-images 1 --workers 0`, rồi `.venv/bin/python -m sed_r2s100k evaluate --output runs/local_smoke --max-images 1`. Mỗi ảnh được lưu thành hai PNG ở kích thước gốc: `predictions/` là mask ID một kênh 0–14 để tính metric, `colored_predictions/` là mask màu RGB theo palette R2S100K để xem trực tiếp. Resume tự kiểm tra fingerprint checkpoint, vocabulary, manifest, model config và mask ID; nếu thiếu mask màu, pipeline tạo lại từ mask ID. Dùng output mới khi thay đổi bất kỳ thành phần nào.

`evaluation/metrics.json` chứa mIoU 15 lớp, mIoU 14 lớp foreground, pixel accuracy, binary foreground IoU/Dice và điểm từng lớp. `per_class.csv` có cả số pixel dự đoán cho từng lớp. Nếu ảnh có background trong GT nhưng model không dự đoán pixel background nào, báo cáo sẽ ghi `quality_warnings`. `confusion_matrix.csv`, 24 panel minh họa và `run_metadata.json` là các đầu ra còn lại. Không có ngưỡng mIoU được áp đặt trước. Mốc 3 giờ chỉ được xác nhận sau benchmark và chạy thật trên RTX 3090; thời gian cài đặt/tải checkpoint không tính trong ngân sách.

## Kiểm thử

```bash
python3 -m pytest -q tests/test_pipeline.py
```

Máy phát triển có RTX 3050 4 GB và đã chạy smoke test SED-L trên một ảnh. Tốc độ RTX 3090 và giới hạn 3 giờ vẫn cần xác nhận bằng benchmark trên server.

## Background và phạm vi mask

`taxonomy.py` định nghĩa đủ ID 0–14. ID 0 (RGB đen) là vùng ngoài các bề mặt đường được gán nhãn; các ID 1–14 là các loại vùng đường cần phân đoạn. Sky, sườn núi, cây, công trình và tường ngoài vùng đường là ví dụ background. Một vật thể nằm trên đường cũng có thể là ID 0 nếu không thuộc 14 lớp vùng đường; thực vật ven đường được gán nhãn Vegetation Misc vẫn là ID 7. Mask màu xuất ID 0 thành đen, mask ID vẫn giữ giá trị 0 để đánh giá.

Nguồn: `../r2s100k/utils/helpers.py` quy định `bg` là ID 0, RGB `(0, 0, 0)`; Bảng I của `../r2s100k.pdf` mô tả 14 lớp đích; mục III-C1 nêu sky làm ví dụ background khi trích ROI. Ví dụ ROI này không phải định nghĩa đầy đủ của nhãn segmentation. Bảng I loại trees khỏi Vegetation Misc. Tài liệu không liệt kê đầy đủ mọi vật thể background.

Vì vậy mô tả background trong project là diễn giải từ phạm vi nhãn: sky và trees có cơ sở từ văn bản; buildings, vehicles và pedestrians là các ví dụ prompt bổ sung suy ra từ phạm vi 14 lớp, không phải danh sách được tác giả công bố. Không gộp cỏ/thực vật ven đường (ID 7), vật rơi trên đường (ID 12) hoặc các bề mặt đường nguy hiểm vào background.

Model nhận 14 prompt vùng đường, sau đó là các prompt cảnh cụ thể ngoài vùng đường như `sky`, `tree`, `mountain hillside`, `building`, `wall beside a road`. SED dự đoán từng kênh riêng; pipeline lấy argmax rồi gộp các kênh cảnh này về ID 0. Cách này theo nguyên tắc evaluator VOC-b của SED, thay vì lấy trung bình các đối tượng không liên quan thành một embedding background. `BACKGROUND_DESCRIPTION` giải thích phạm vi; CLIP chỉ nhận các cụm ngắn trong `DESCRIPTION_PROMPTS` và `BACKGROUND_PROMPT_VARIANTS`. Trước inference hoặc benchmark, CLI tạo `runs/<output>/vocabulary.json` từ taxonomy; `run_metadata.json` lưu cả ánh xạ kênh SED → ID R2S100K.

Để đổi prompt, sửa `DESCRIPTION_PROMPTS` cho ID 1–14 hoặc `BACKGROUND_PROMPT_VARIANTS` cho các kênh gộp về ID 0. Sau đó dùng output mới, ví dụ `runs/local_background_v1`: pipeline từ chối tái sử dụng output có vocabulary khác. Mô tả rõ hơn không bảo đảm tăng IoU: cần so sánh cùng tập ảnh với cấu hình cũ trước khi kết luận chất lượng.
# SED
