from django.shortcuts import render
from django.http import HttpResponse
from .forms import ItineraryForm
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import re
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import login, logout, authenticate, get_user_model
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse

# <=================== Signup View ============================>

def signup_view(request):
    if request.method == "POST":
        username = request.POST["username"]
        email = request.POST["email"]
        password = request.POST["password"]
        confirmpassword = request.POST["confirmpassword"]

        if password == confirmpassword:
            if not User.objects.filter(email=email).exists():
                user = User(username=username, email=email)
                user.set_password(password)  # Properly hash the password
                user.save()
                login(request, user)
                return redirect("/")
            else:
                messages.error(request, "Email already exists.")
                return redirect("/signup/")
        else:
            messages.error(request, "Passwords do not match.")
            return redirect("/signup/")
    else:
        return render(request, "itinerary/signup.html")



#===================== Signin View =======================
def signin_view(request):
    if request.method == "POST":
        email = request.POST['email']
        password = request.POST['password']

        try:
            # Check if the user with the provided email exists
            user_obj = User.objects.get(email=email)
            username = user_obj.username  # Get the associated username
        except User.DoesNotExist:
            # If no user with that email exists, show an error message and trigger the signup modal
            messages.error(request, "You are not registered, Please register first.")
            return render(request, "itinerary/index.html", {'signup_error': True})

        # Authenticate the user with the username and password
        user = authenticate(request, username=username, password=password)

        if user is not None:
            # Log the user in
            login(request, user)
            return redirect('/')
        else:
            # If authentication fails (wrong password), show an error and trigger login modal
            messages.error(request, "Invalid email or password.")
            return render(request, "itinerary/index.html", {'login_error': True})

    return redirect('/')  # Fallback for non-POST requests


#=========================== Get Available Cities ==========================#

def get_available_cities(request):
    file_path = "tourdata.xlsx"
    try:
        data = pd.read_excel(file_path, sheet_name="Sheet1")
    except Exception as e:
        return JsonResponse({"error": f"Error loading data. {str(e)}"}, status=500)

    # Extract unique cities
    cities = data["City Name"].dropna().unique().tolist()

    # Return cities in JSON format
    return JsonResponse(cities, safe=False)


# Signout
def signout_view(request):
    logout(request)
    return redirect("/")


# Initialize geocoder
geolocator = Nominatim(user_agent="tour_planner")


def get_coordinates(city_name):
    """
    Retrieves the latitude and longitude of a given city.
    Uses geopy's Nominatim service to geocode the city name.
    """
    location = geolocator.geocode(city_name)
    if location:
        return location.latitude, location.longitude  # type: ignore
    return None, None


def calculate_distance(loc1, loc2):
    """
    Calculates the geodesic distance between two locations.
    Each location is a dictionary with 'latitude' and 'longitude' keys.
    Returns the distance in kilometers.
    """
    try:
        return geodesic(
            (loc1["latitude"], loc1["longitude"]), (loc2["latitude"], loc2["longitude"])
        ).km
    except Exception:
        return np.nan


def to_title_case(s):
    """
    Converts a string to title case.
    """
    return s.title()


def find_nearest_places(latitude, longitude, places_df, max_distance_km=50):
    """
    Finds the nearest places within a specified distance from a given latitude and longitude.
    """
    places_df["Latitude"] = pd.to_numeric(places_df["Latitude"], errors="coerce")
    places_df["Longitude"] = pd.to_numeric(places_df["Longitude"], errors="coerce")
    places_df = places_df.dropna(subset=["Latitude", "Longitude"])

    places_df["distance"] = places_df.apply(
        lambda row: geodesic(
            (latitude, longitude), (row["Latitude"], row["Longitude"])
        ).kilometers,
        axis=1,
    )

    nearest_places = places_df[places_df["distance"] <= max_distance_km].sort_values(
        "distance"
    )
    return nearest_places


def format_duration(hours):
    """
    Converts hours in decimal form to a human-readable format like '1 hour 46 minutes'.
    """
    hr = int(hours)
    minutes = round((hours - hr) * 60)
    formatted = f"{hr} hour{'s' if hr != 1 else ''}" if hr > 0 else ""
    if minutes > 0:
        if formatted:
            formatted += " "
        formatted += f"{minutes} minute{'s' if minutes != 1 else ''}"
    return formatted


def itinerary_view(request):
    file_path = "tourdata.xlsx"
    try:
        data = pd.read_excel(file_path, sheet_name="Sheet1")
    except Exception as e:
        return HttpResponse(
            f"Error loading data. Please ensure the data file is available. Error: {str(e)}"
        )
    data.rename(columns={"\ufeffMain Category": "MainCategory"}, inplace=True)

    # Extract unique cities
    cities = data["City Name"].dropna().unique().tolist()

    if request.method == "POST":
        form = ItineraryForm(request.POST)
        if form.is_valid():
            user_start_city = form.cleaned_data["start_city"]
            user_trip_days = form.cleaned_data["trip_days"]
            user_preferences = form.cleaned_data["preferences"]

            # Store user_start_city in the session to pass it to the next view
            request.session["user_start_city"] = user_start_city

            user_preferences = (
                to_title_case(user_preferences) if user_preferences else ""
            )
            user_start_city = to_title_case(user_start_city)

            file_path = "tourdata.xlsx"
            try:
                data = pd.read_excel(file_path, sheet_name="Sheet1")
            except Exception as e:
                return HttpResponse(
                    f"Error loading data. Please ensure the data file is available. Error: {str(e)}"
                )
            data.rename(columns={"\ufeffMain Category": "MainCategory"}, inplace=True)

            # Extract unique cities
            cities = data["City Name"].dropna().unique().tolist()

            imputer = SimpleImputer(strategy="mean")
            if data[["Latitude", "Longitude"]].isna().any().any():
                lat_lon_data = data[["Latitude", "Longitude"]]
                lat_lon_imputed = imputer.fit_transform(lat_lon_data)
                data[["Latitude", "Longitude"]] = lat_lon_imputed

            def extract_duration(timing_str):
                if isinstance(timing_str, str):
                    numbers = re.findall(r"\d+", timing_str)
                    if numbers:
                        numbers = list(map(int, numbers))
                        if len(numbers) == 2:
                            return (numbers[0] + numbers[1]) / 2
                        elif len(numbers) == 1:
                            return numbers[0]
                return np.nan

            data["Suggested Duration (Hours)"] = data["Suggested Timing"].apply(
                extract_duration
            )
            data.dropna(subset=["Suggested Duration (Hours)"], inplace=True)

            features = ["Latitude", "Longitude"]
            target = "Suggested Duration (Hours)"

            X = data[features].copy()
            y = data[target].copy()

            X.fillna(X.mean(), inplace=True)
            y.fillna(y.mean(), inplace=True)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)

            data["Predicted Duration (Hours)"] = model.predict(X[features])

            user_start_lat, user_start_lon = get_coordinates(user_start_city)
            if user_start_lat is None or user_start_lon is None:
                return HttpResponse(
                    f"Could not find coordinates for the city: {user_start_city}"
                )

            user_start_location = {
                "latitude": user_start_lat,
                "longitude": user_start_lon,
            }

            filtered_data = data[
                (data["MainCategory"] == user_preferences)
                & (data["City Name"] == user_start_city)
            ].copy()

            if filtered_data.empty:
                filtered_data = data[data["City Name"] == user_start_city].copy()
                message = f"There are no {user_preferences.lower()} in {user_start_city}. Showing all available places instead."
                print("MESSAGE", message)
            else:
                message = ""

            if not filtered_data.empty:
                filtered_data["Distance From Start (km)"] = filtered_data.apply(
                    lambda row: calculate_distance(
                        user_start_location,
                        {"latitude": row["Latitude"], "longitude": row["Longitude"]},
                    ),
                    axis=1,
                )

                filtered_data.dropna(subset=["Distance From Start (km)"], inplace=True)
                filtered_data.sort_values(by="Distance From Start (km)", inplace=True)

                itinerary = {day: [] for day in range(1, user_trip_days + 1)}
                day_hours = 8
                current_location = user_start_location

                for day in range(1, user_trip_days + 1):
                    remaining_hours = day_hours
                    daily_places = []

                    for index, row in filtered_data.iterrows():
                        duration = row["Predicted Duration (Hours)"]
                        if duration <= remaining_hours:
                            image_urls = (
                                row["Image URL"].split(",")
                                if pd.notna(row["Image URL"])
                                else []
                            )
                            daily_places.append(
                                {
                                    "place_name": row["Place Name"],
                                    "predicted_duration_hours": duration,
                                    "current_location": current_location,
                                    "next_location": {
                                        "latitude": row["Latitude"],
                                        "longitude": row["Longitude"],
                                    },
                                    "distance_from_current": row[
                                        "Distance From Start (km)"
                                    ],
                                    "main_category": row["MainCategory"],
                                    "subcategory": row["Subcategory"],
                                    "link": row["Link"],
                                    "overview_content": row["Overview Content"],
                                    "additional_timings": row["Additional Timings"],
                                    "suggested_timing": row["Suggested Timing"],
                                    "tips": row["Tips"],
                                    "image_urls": image_urls,
                                    "latitude": row["Latitude"],
                                    "longitude": row["Longitude"],
                                }
                            )

                            remaining_hours -= duration
                            current_location = {
                                "latitude": row["Latitude"],
                                "longitude": row["Longitude"],
                            }
                            filtered_data = filtered_data.drop(index)

                            filtered_data["Distance From Start (km)"] = (
                                filtered_data.apply(
                                    lambda r: calculate_distance(
                                        current_location,
                                        {
                                            "latitude": r["Latitude"],
                                            "longitude": r["Longitude"],
                                        },
                                    ),
                                    axis=1,
                                )
                            )
                            filtered_data.dropna(
                                subset=["Distance From Start (km)"], inplace=True
                            )
                            filtered_data.sort_values(
                                by="Distance From Start (km)", inplace=True
                            )

                            if remaining_hours <= 0:
                                break

                    itinerary[day] = daily_places
                    city = request.session["user_start_city"]

                formatted_itinerary = {
                    day: [
                        {
                            "place_name": place["place_name"],
                            "predicted_duration_hours": format_duration(
                                place["predicted_duration_hours"]
                            ),
                            "current_location": place["current_location"],
                            "next_location": place["next_location"],
                            "distance_from_current": place["distance_from_current"],
                            "main_category": place["main_category"],
                            "subcategory": place["subcategory"],
                            "link": place["link"],
                            "overview_content": place["overview_content"],
                            "additional_timings": place["additional_timings"],
                            "suggested_timing": place["suggested_timing"],
                            "tips": place["tips"],
                            "image_urls": place["image_urls"],
                            "latitude": place["latitude"],
                            "longitude": place["longitude"],
                        }
                        for place in places
                    ]
                    for day, places in itinerary.items()
                }

                last_location = None
                for day in sorted(formatted_itinerary.keys(), reverse=True):
                    if formatted_itinerary[day]:
                        last_location = formatted_itinerary[day][-1]["next_location"]
                        break

                nearest_places = []
                if not filtered_data.empty and last_location:
                    filtered_data["Distance_From_Last_Location_km"] = (
                        filtered_data.apply(
                            lambda row: calculate_distance(
                                last_location,
                                {
                                    "latitude": row["Latitude"],
                                    "longitude": row["Longitude"],
                                },
                            ),
                            axis=1,
                        )
                    )
                    filtered_data.dropna(
                        subset=["Distance_From_Last_Location_km"], inplace=True
                    )
                    nearest_places = filtered_data.sort_values(
                        by="Distance_From_Last_Location_km"
                    ).head(5)

                # Rename columns for template compatibility
                nearest_places_list = nearest_places.rename( # type: ignore
                    columns={  # type: ignore
                        "Distance_From_Last_Location_km": "distance_from_current",
                        "Place Name": "place_name",
                        "MainCategory": "main_category",
                        "Subcategory": "subcategory",
                        "Link": "link",
                        "Overview Content": "overview_content",
                        "Latitude": "latitude",
                        "Longitude": "longitude",
                        "Image URL": "image_urls",
                        "Predicted Duration (Hours)": "predicted_duration_hours",
                    }
                ).to_dict(orient="records")

                # Ensure image_urls is a list
                for place in nearest_places_list:
                    if isinstance(place["image_urls"], str):
                        place["image_urls"] = place["image_urls"].split(
                            ","
                        ) 

            else:
                formatted_itinerary = {}
                nearest_places_list = []

            if user_start_city not in data["City Name"].values:  # type: ignore
                error_message = (
                    f"Sorry, we couldn't find any data for the city: {user_start_city}."
                )
                return render(
                    request,
                    "itinerary/index.html",
                    {"form": form, "error_message": error_message},
                )

            # Store user_start_city in the session to pass it to the next view
            context = {
                "itinerary": formatted_itinerary,
                "nearest_places": nearest_places_list,
                "message": message,
                "form": form,
                "city": city,
                "cities": cities,
            }
            return render(request, "itinerary/results.html", context)
    else:
        form = ItineraryForm()
        cities = data["City Name"].dropna().unique().tolist()

    return render(request, "itinerary/index.html", {"form": form, "cities": cities})



######################################### CHATBOT############################

import re
import json
import google.generativeai as genai
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import speech_recognition as sr
from django.contrib.auth.decorators import login_required
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) # type: ignore

# Set your API key directly (ensure this is handled securely in production)
api_key = "AIzaSyCuT9iFU9UlsrwEwN0ip8-TkgnE41-BR8g"

# Configure the API client
genai.configure(api_key=api_key)

# Create a chat model
model_name = "gemini-1.5-flash"
model = genai.GenerativeModel(model_name)

# Basic text correction function
def basic_text_correction(text):
    text = re.sub(r'[*]+', '', text)  # Remove unwanted special characters
    text = re.sub(r'\s*([.,!?;:])\s*', r'\1 ', text)  # Ensure proper spacing around punctuation
    text = '. '.join([sentence.capitalize() for sentence in text.split('. ')])  # Capitalize sentences
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space and trim spaces
    return text

def check_authentication(request):
    is_authenticated = request.user.is_authenticated
    return JsonResponse({'is_authenticated': is_authenticated})

# Ensure the chat history is stored per user session
def get_user_chat_session(request):
    if 'chat_history' not in request.session:
        request.session['chat_history'] = [
            {"role": "user", "parts": "Hello"},
            {"role": "model", "parts": "Hi there! How can I assist you today?"}
        ]
    return request.session['chat_history']

@login_required(login_url='/signin')
@csrf_exempt
def chat_with_user(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_input = data.get('message', '')

            logger.info(f"User input received: {user_input}")

            creation_patterns = [
                r'\bwho\s+(made|created|developed|is)\s+you\b',
                r'\bwhere\s+did\s+you\s+come\s+from\b',
                r'\bwho\s+is\s+your\s+creator\b'
            ]
            
            if any(re.search(pattern, user_input, re.IGNORECASE) for pattern in creation_patterns):
                response_text = "I am a Pathintel production but my origin is from Google, as I am trained by Google."
                return JsonResponse({"response": response_text})

            # Retrieve the user's chat history from the session
            user_chat_history = get_user_chat_session(request)
            
            # Start a new chat for each user session
            chat = model.start_chat(history=user_chat_history)

            response = chat.send_message(f"{user_input}. Please keep your response brief.")

            logger.info(f"Raw response from chat: {response}")

            cleaned_response = basic_text_correction(response.text)
            logger.info(f"Cleaned response: {cleaned_response}")

            user_chat_history.append({"role": "user", "parts": user_input})
            user_chat_history.append({"role": "model", "parts": cleaned_response})

            # Update the session's chat history
            request.session['chat_history'] = user_chat_history

            if hasattr(response, 'safety_ratings') and response.safety_ratings: # type: ignore
                for rating in response.safety_ratings: # type: ignore
                    if rating.category in [
                        'HARM_CATEGORY_SEXUALLY_EXPLICIT', 
                        'HARM_CATEGORY_HATE_SPEECH', 
                        'HARM_CATEGORY_HARASSMENT', 
                        'HARM_CATEGORY_DANGEROUS_CONTENT'
                    ]:
                        logger.warning(f"Safety filter triggered: {rating.category}")
                        return JsonResponse({
                            "response": "Sorry, your request triggered a safety filter. Please try again with a different input."
                        }, status=200)

            return JsonResponse({"response": cleaned_response})

        except json.JSONDecodeError:
            logger.error("Invalid JSON format")
            return JsonResponse({"error": "Invalid JSON format."}, status=400)
        except AttributeError:
            logger.error("Error: The response object does not have the 'safety_ratings' attribute")
            return JsonResponse({"response": "We are not supposed to answer this."}, status=200)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return JsonResponse({"error": "An unexpected error occurred. Please try again later."}, status=500)

    return JsonResponse({"error": "Only POST method is allowed."}, status=405)

@login_required(login_url='/signin')
@csrf_exempt
def get_suggested_questions(request):
    if request.method == 'GET':
        try:
            heading = request.session.get('user_start_city', 'DefaultCity')

            # Retrieve the user's chat history
            user_chat_history = get_user_chat_session(request)

            chat = model.start_chat(history=user_chat_history)
            response = chat.send_message(f"Please generate 3 suggested questions related to '{heading}'. Keep the questions short and concise, and only provide the questions.")
            suggested_questions = response.text.strip().split('\n')

            suggested_questions = [q for q in suggested_questions if q.strip()][:3]

            return JsonResponse({"questions": suggested_questions})
        except Exception as e:
            logger.error(f"Error generating suggested questions: {e}")
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Only GET method is allowed."}, status=405)

def chat_page(request):
    return render(request, 'itinerary/chat.html')

@login_required(login_url='/signin')
@csrf_exempt
def transcribe_audio(request):
    if request.method == 'POST':
        if 'audio' in request.FILES:
            audio_file = request.FILES['audio']
            file_name = default_storage.save('temp_audio.wav', ContentFile(audio_file.read()))
            file_path = default_storage.path(file_name)

            recognizer = sr.Recognizer()
            with sr.AudioFile(file_path) as source:
                audio = recognizer.record(source)
                try:
                    transcript = recognizer.recognize_google(audio) # type: ignore
                except sr.UnknownValueError:
                    transcript = 'Unable to understand audio'
                except sr.RequestError as e:
                    transcript = f'Error with the request: {e}'

            default_storage.delete(file_name)

            return JsonResponse({'transcript': transcript})
        else:
            return JsonResponse({'error': 'No audio file provided'}, status=400)

    return JsonResponse({'error': 'Invalid request method'},status=405)

    ###### Team Page ######


###### Team Page ######


def team_view(request):
    return render(request, "itinerary/team.html")


###### Contact Us Page ######


def contact_view(request):
    return render(request, "itinerary/contact.html")
