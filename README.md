# Graph Building — Company Knowledge Graph

Xây dựng knowledge graph từ dữ liệu đăng ký doanh nghiệp Việt Nam và học vector embedding cho từng công ty bằng HinSAGE. Embedding có thể dùng làm features bổ sung cho downstream task như credit scoring hay fraud detection — nắm bắt tín hiệu rủi ro từ quan hệ giữa các công ty (cùng người đại diện, cùng ngành, cùng địa chỉ) thay vì chỉ dùng số liệu tài chính.

---

## Cấu trúc project

```
Graph building/
├── notebook/
│   ├── EDA.ipynb               # phân tích dữ liệu, chuẩn bị cho build graph
│   ├── build_graph.ipynb       # xây dựng heterogeneous graph
│   └── model.ipynb             # train HinSAGE, lấy embedding
├── data/
│   ├── company_info_sample_1000.csv   # dữ liệu gốc 1000 doanh nghiệp
│   ├── df_processed.csv / .pkl        # dữ liệu sau EDA
│   ├── nodes.csv / edges.csv          # nodes và edges của graph
│   ├── company_graph.graphml          # graph xuất ra GraphML
│   ├── company_embeddings_lp.csv      # embedding từ Link Prediction
│   └── company_embeddings_nc.csv      # embedding từ Node Classification
└── photo/
    ├── graph_demo.png / .html
    ├── embedding_tsne_lp.png
    ├── embedding_tsne_nc.png
    └── embedding_quality_check.png
```

---

## Cài đặt

```bash
pip install stellargraph networkx pandas numpy scikit-learn matplotlib seaborn
```
---

## Cách chạy

Chạy theo thứ tự:

```
1. notebook/EDA.ipynb
2. notebook/build_graph.ipynb
3. notebook/model.ipynb
```

Mỗi notebook đọc output của notebook trước. Kết quả cuối cùng là 2 file embedding tại `data/`.

---

## Kết quả

**Graph:** 2661 nodes (Company, Person, Industry, Province, Ward), 4369 edges (DAI_DIEN, LINH_VUC, TAI_PROVINCE, TAI_WARD, CHI_NHANH).

**Embedding:** Mỗi công ty được biểu diễn bằng vector 32 chiều. Đánh giá chất lượng bằng cosine similarity:

| So sánh | Similarity |
|---|---|
| Cùng ngành nghề | 0.659 |
| Khác ngành nghề | 0.599 |
| Cùng người đại diện | 0.756 |
| Khác người đại diện | 0.378 |

Embedding ghi nhận tốt cấu trúc quan hệ, đặc biệt qua cạnh DAI_DIEN — cùng người đại diện thì vector gần nhau hơn rõ rệt.

---

## Hạn chế

- 1000 công ty quá nhỏ để train GNN có ý nghĩa thống kê
- Proxy label (active/inactive) mất cân bằng nghiêm trọng (6/784), node classification chưa đáng tin cậy
- Industry, Province, Ward dùng random features vì không có attribute thực
- Chưa có dữ liệu quan hệ sở hữu và chuỗi cung ứng
