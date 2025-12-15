from src.predict import predict_yield

params = [
    dict(crop='Rice', district='Colombo', year=2024, rainfall=1500, temperature=27, humidity=75),
    dict(crop='Rice', district='Colombo', year=2024, rainfall=1000, temperature=27, humidity=75),
    dict(crop='Tea', district='Kandy', year=2024, rainfall=1500, temperature=27, humidity=75),
    dict(crop='Rice', district='Colombo', year=2025, rainfall=1500, temperature=30, humidity=80),
]

for p in params:
    pred = predict_yield(**p)
    print(p, '=>', pred)
