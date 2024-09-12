from django import forms

class ItineraryForm(forms.Form):
    start_city = forms.CharField(label='Start City', max_length=100)
    trip_days = forms.IntegerField(label='Number of Days')
    preferences = forms.CharField(label='Preferences', required=False, help_text="Enter preferences separated by commas (e.g., Beach, Museum)")
