import requests

CLIENT_ID = "qym6nfg7E20fyAtpWCG0kE0jF9zG59h3"
CLIENT_SECRET = "UzGhJzXQaGiOqgDt"


def get_access_token():
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        token_data = response.json()
        return token_data["access_token"]
    else:
        print("Error: Unable to fetch access token")
        return None


def get_iata_code(city_name):
    access_token = get_access_token()
    if not access_token:
        return None

    url = f"https://test.api.amadeus.com/v1/reference-data/locations?subType=CITY&keyword={city_name}"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json().get("data", [])
        if data:
            return data[0]['iataCode']
    return None
def search_flights(origin, destination, date, adults):
    origin_code = get_iata_code(origin)
    destination_code = get_iata_code(destination)
    

    if not origin_code or not destination_code:
        return "Unable to find IATA codes for origin or destination."

    access_token = get_access_token()
    if not access_token:
        return "Unable to get access token"

    url = f"https://test.api.amadeus.com/v2/shopping/flight-offers?originLocationCode={origin_code}&destinationLocationCode={destination_code}&departureDate={date}&adults={adults}"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    response = requests.get(url, headers=headers)
   

    if response.status_code == 200:
        flights = response.json().get("data", [])

        flight_details = []
        if flights:
            
            for flight in flights:
                segments = flight['itineraries'][0]['segments']
                departure = segments[0]['departure']['at']
                arrival = segments[-1]['arrival']['at']
                aircraft_names = []

        
                for segment in segments:
           
                    aircraft_code = segment.get('aircraft', {}).get('code', 'Unknown')
                    aircraft_names.append(aircraft_code)

                aircraft_names_str = ", ".join(aircraft_names)
                currency = flight['price']['currency']
                total_price = flight['price']['total']
    
                flight_details.append([
            f"Flight from {origin} to {destination}",
            f"Departure: {departure}",
            f"Arrival: {arrival}",
            f"Aircraft: {aircraft_names_str}",
            f"Price: {total_price} {currency}"
        ])

        
        return flight_details
    else:
        return f"Error: {response.status_code} - {response.text}"
