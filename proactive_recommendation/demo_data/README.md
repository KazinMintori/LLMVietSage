# VietSage proactive demo data

Bo du lieu nay la du lieu gia lap. Moi ten khach san, phong, khach luu tru,
gia dich vu va dia diem co nhan `DEMO` deu khong dai dien cho mot giao dich
thuc te.

## Muc tieu

- Kiem thu RAG voi thong tin khach san, dich vu va diem den co dap an ro rang.
- Kiem thu proactive recommendation theo moc thoi gian bang simulated clock.
- Kiem thu consent o ba trang thai: granted, denied va withdrawn.
- Kiem thu fallback khi khong co GPS, khong co push permission hoac thieu data.
- Kiem thu idempotency va frequency cap truoc khi ket noi PMS that.

## Tep chinh

- `vietsage_demo_source.json`: source of truth co cau truc.
- `expected_proactive_cases.json`: test oracle cho rule engine.
- `../output/pdf/VSDEMO 0001-26.pdf`: nguon RAG sinh tu JSON.
- `manifest.json`: mapping nguon PDF demo, nam cung bo du lieu.

## Tao lai PDF va kiem tra

```powershell
python proactive_recommendation/scripts/generate_demo_source_pdf.py
python proactive_recommendation/scripts/validate_demo_data.py
```

Script sinh PDF can thu vien `reportlab` (`python -m pip install reportlab`).

Manifest co the dung voi adapter ingestion RAG ben ngoai khi adapter do san sang. Pipeline ingestion legacy cua project phu thuoc database da go; khong can pipeline do de chay nghien cuu proactive hoac kiem tra du lieu demo.
