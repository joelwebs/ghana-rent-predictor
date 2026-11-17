
from src.predict import predict_price


properties = [
    {
        "listing_type": "rentals",
        "category": "Detached",
        "bedrooms": 5,
        "bathrooms": 3,
        "floor_area": 120,
        "location": "Madina",
        "condition": "New",
        "region": "Greater Accra",
        "locality": "Accra",
        "parking_space": 1,
        "is_furnished": "Furnished",
        "lat": 5.6037,
        "lng": -0.1870,
        "amenities_count": 5,
    },
    {
        "listing_type": "rentals",
        "category": "Flats",
        "bedrooms": 3,
        "bathrooms": 3,
        "floor_area": 200,
        "location": "Madina",
        "condition": "Good",
        "region": "Greater Accra",
        "locality": "Accra",
        "parking_space": 0,
        "is_furnished": "Unfurnished",
        "lat": 5.6680,
        "lng": -0.1640,
        "amenities_count": 2,
    },
    {
        "listing_type": "rentals",
        "category": "Detached",
        "bedrooms": 1,
        "bathrooms": 1,
        "floor_area": 100,
        "location": "East Legon",
        "condition": "New",
        "region": "Greater Accra",
        "locality": "Accra",
        "parking_space": 0,
        "is_furnished": "Semi-Furnished",
        "lat": 5.7050,
        "lng": -0.1650,
        "amenities_count": 2,
    },
]


for index, property_data in enumerate(properties, start=1):
    prediction = predict_price(property_data)

    print(f"\nProperty {index}")
    print("------------------------")
    print(f"Category: {property_data['category']}")
    print(f"Bedrooms: {property_data['bedrooms']}")
    print(f"Location: {property_data['location']}")
    print(f"Predicted listed price: GHS {prediction:,.2f}")
