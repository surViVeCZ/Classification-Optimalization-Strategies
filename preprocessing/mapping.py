from pandas import DataFrame
import numpy as np

continents = {
    'North America': ['Canada', 'United States', 'United States of America', 'Mexico', 'Bermuda'],
    'South America': ['Brazil', 'Argentina', 'Peru', 'Chile', 'Colombia', 'Ecuador', 'Venezuela', 'Bolivia', 'Paraguay', 'Suriname', 'Uruguay'],
    'Europe': ['United Kingdom', 'France', 'Italy', 'Germany', 'Spain', 'Ukraine', 'Poland', 'Romania', 'Netherlands', 'Belgium', 'Greece', 'Portugal', 'Czechia', 'Czech Republic' 'Hungary', 'Sweden', 'Austria', 'Switzerland', 'Bulgaria', 'Denmark', 'Finland', 'Norway', 'Ireland', 'Croatia', 'Slovakia', 'Lithuania', 'Slovenia', 'Latvia', 'Estonia', 'Luxembourg', 'Malta', 'Iceland', 'Jersey', 'Isle of Man', 'Monaco', 'Liechtenstein'],
    'Africa': ['Nigeria', 'Ethiopia', 'Egypt', 'Democratic Republic of the Congo', 'Tanzania', 'South Africa', 'Kenya', 'Uganda', 'Algeria', 'Sudan', 'Morocco', 'Angola', 'Mozambique', 'Ghana', 'Madagascar', 'Cameroon', 'Côte d’Ivoire', 'Niger', 'Burkina Faso', 'Mali', 'Malawi', 'Zambia', 'Somalia', 'Senegal', 'Chad', 'Zimbabwe', 'Guinea', 'Rwanda', 'Benin', 'Tunisia', 'Burundi', 'South Sudan', 'Togo', 'Sierra Leone', 'Libya', 'Central African Republic', 'Eritrea', 'Namibia', 'Gambia', 'Botswana', 'Gabon', 'Lesotho', 'Guinea-Bissau', 'Equatorial Guinea', 'Mauritania', 'Eswatini', 'Djibouti', 'Comoros', 'Cape Verde', 'São Tomé and Príncipe'],
    'Asia': ['China', 'India', 'Indonesia', 'Pakistan', 'Bangladesh', 'Japan', 'Philippines', 'Vietnam', 'Turkey', 'Iran', 'Thailand', 'Myanmar', 'South Korea', 'Iraq', 'Afghanistan', 'Saudi Arabia', 'Uzbekistan', 'Malaysia', 'Nepal', 'Yemen', 'North Korea', 'Taiwan', 'Syria', 'Sri Lanka', 'Kazakhstan', 'Cambodia', 'Azerbaijan', 'United Arab Emirates', 'Tajikistan', 'Israel', 'Laos', 'Kyrgyzstan', 'Jordan', 'Lebanon', 'Singapore', 'Oman', 'Palestine', 'Kuwait', 'Georgia', 'Mongolia', 'Armenia', 'Qatar', 'Bahrain', 'Timor-Leste', 'Cyprus', 'Bhutan', 'Maldives', 'Brunei'],
    'Oceania': ['Australia', 'Papua New Guinea', 'New Zealand', 'Fiji', 'Solomon Islands', 'Vanuatu', 'New Caledonia', 'French Polynesia', 'Samoa', 'Guam', 'Kiribati']
}

continent_ids = {
    'Unknown': 0,
    'North America': 1,
    'South America': 2,
    'Europe': 3,
    'Africa': 4,
    'Asia': 5,
    'Oceania': 6
}

country_ids = {
    "Afghanistan": 1, "Albania": 2, "Algeria": 3, "Andorra": 4, "Angola": 5, "Antigua and Barbuda": 6, "Argentina": 7,
    "Armenia": 8, "Australia": 9, "Austria": 10, "Azerbaijan": 11, "Bahamas": 12, "Bahrain": 13, "Bangladesh": 14,
    "Barbados": 15, "Belarus": 16, "Belgium": 17, "Belize": 18, "Benin": 19, "Bhutan": 20, "Bolivia": 21, "Bosnia and Herzegovina": 22,
    "Botswana": 23, "Brazil": 24, "Brunei": 25, "Bulgaria": 26, "Burkina Faso": 27, "Burundi": 28, "Côte d'Ivoire": 29,
    "Cabo Verde": 30, "Cambodia": 31, "Cameroon": 32, "Canada": 33, "Central African Republic": 34, "Chad": 35, "Chile": 36,
    "China": 37, "Colombia": 38, "Comoros": 39, "Congo": 40, "Congo-Brazzaville": 40, "Costa Rica": 41, "Croatia": 42, "Cuba": 43,
    "Cyprus": 44, "Czechia": 45, "Czech Republic": 45, "Democratic Republic of the Congo": 46, "Denmark": 47, "Djibouti": 48, "Dominica": 49,
    "Dominican Republic": 50, "Ecuador": 51, "Egypt": 52, "El Salvador": 53, "Equatorial Guinea": 54, "Eritrea": 55, "Estonia": 56,
    "Eswatini": 57, "Swaziland": 57, "Ethiopia": 58, "Fiji": 59, "Finland": 60, "France": 61, "Gabon": 62, "Gambia": 63, "Georgia": 64,
    "Germany": 65, "Ghana": 66, "Greece": 67, "Grenada": 68, "Guatemala": 69, "Guinea": 70, "Guinea-Bissau": 71, "Guyana": 72, "Haiti": 73,
    "Holy See": 74, "Honduras": 75, "Hungary": 76, "Iceland": 77, "India": 78, "Indonesia": 79, "Iran": 80, "Iraq": 81, "Ireland": 82,
    "Israel": 83, "Italy": 84, "Jamaica": 85, "Japan": 86, "Jordan": 87, "Kazakhstan": 88, "Kenya": 89, "Kiribati": 90, "Kuwait": 91,
    "Kyrgyzstan": 92, "Laos": 93, "Latvia": 94, "Lebanon": 95, "Lesotho": 96, "Liberia": 97, "Libya": 98, "Liechtenstein": 99, "Lithuania": 100,
    "Luxembourg": 101, "Madagascar": 102, "Malawi": 103, "Malaysia": 104, "Maldives": 105, "Mali": 106, "Malta": 107, "Marshall Islands": 108,
    "Mauritania": 109, "Mauritius": 110, "Mexico": 111, "Micronesia": 112, "Moldova": 113, "Monaco": 114, "Mongolia": 115, "Montenegro": 116,
    "Morocco": 117, "Mozambique": 118, "Myanmar": 119, "Burma": 119, "Namibia": 120, "Nauru": 121, "Nepal": 122, "Netherlands": 123,
    "New Zealand": 124, "Nicaragua": 125, "Niger": 126, "Nigeria": 127, "North Korea": 128, "North Macedonia": 129,"Norway": 130, "Oman": 131,
    "Pakistan": 132, "Palau": 133, "Palestine State": 134, "Panama": 135, "Papua New Guinea": 136, "Paraguay": 137, "Peru": 138, "Philippines": 139,
    "Poland": 140, "Portugal": 141, "Qatar": 142, "Romania": 143, "Russia": 144, "Rwanda": 145, "Saint Kitts and Nevis": 146, "Saint Lucia": 147,
    "Saint Vincent and the Grenadines": 148, "Samoa": 149, "San Marino": 150, "Sao Tome and Principe": 151, "Saudi Arabia": 152, "Senegal": 153,
    "Serbia": 154, "Seychelles": 155, "Sierra Leone": 156, "Singapore": 157, "Slovakia": 158, "Slovenia": 159, "Solomon Islands": 160, "Somalia": 161,
    "South Africa": 162, "South Korea": 163, "South Sudan": 164, "Spain": 165, "Sri Lanka": 166, "Sudan": 167, "Suriname": 168, "Sweden": 169,
    "Switzerland": 170, "Syria": 171, "Tajikistan": 172, "Tanzania": 173, "Thailand": 174, "Timor-Leste": 175, "Togo": 176, "Tonga": 177,
    "Trinidad and Tobago": 178, "Tunisia": 179, "Turkey": 180, "Turkmenistan": 181, "Tuvalu": 182, "Uganda": 183, "Ukraine": 184,
    "United Arab Emirates": 185, "United Kingdom": 186, "United States of America": 187, "United States": 187, "Uruguay": 188, "Uzbekistan": 189, "Vanuatu": 190,
    "Venezuela": 191, "Vietnam": 192, "Yemen": 193, "Zambia": 194, "Zimbabwe": 195,
    "Unknown": 300
}

_malicious_domain_top_hosting_countries = [
    187, # USA
    144, # Russia
    37,  # China
    24,  # Brazil
    192, # Vietnam
    184, # Ukraine
    78,  # India
]

def get_continent_name(country_name):
    for continent, countries in continents.items():
        if country_name in countries:
            return continent
    return None

def get_continent_id(country_name):
    try:
        continent_name = get_continent_name(country_name)
    except:
        continent_name = "Unknown"

    if continent_name in continent_ids.keys():
        return continent_ids[continent_name]
    else:
        return 0
    

def hash_continents(countries):
    if not countries:
        return 0

    continents_set = set()
    for country_name in countries:
        try:
            continent_id = get_continent_id(country_name)
            continents_set.add(continent_id)
        except:
            continue

    hash = 0
    continent_ids_count = len(continent_ids)

    for continent_id in continents_set:
        hash += continent_id

        if continent_id > continent_ids_count:
            hash *= 2

    return hash % 2147483647


def get_continent_count(countries):
    if not countries:
        return 0

    continents_set = set()
    for country_name in countries:
        try:
            continent_id = get_continent_id(country_name)
            continents_set.add(continent_id)
        except:
            continue

    return len(continents_set)


def hash_countries(countries):
    if not countries:
        return 0

    country_ids_set = set()
    for country_name in countries:
        country_id = 300  # Unknown
        if country_name in country_ids:
            country_id = country_ids[country_name]
        country_ids_set.add(country_id)

    hash = 0
    country_ids_count = len(country_ids)

    for country_id in country_ids_set:
        hash += country_id

        if hash > country_ids_count:
            hash *= 2

    return hash % 2147483647


def has_malicious_hosting_country(countries):
    if not countries:
        return 0

    country_ids_set = set()
    for country_name in countries:
        if country_name in country_ids:
            country_id = country_ids[country_name]
            if country_id in _malicious_domain_top_hosting_countries:
                return 1
        
    return 0