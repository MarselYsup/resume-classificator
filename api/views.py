from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from api.serializers import ResumeSerializer
import joblib

from api.services import  clean_resume

loaded_model = joblib.load(open('./api/model/model.pkl', 'rb'))
loaded_vect = joblib.load(open('./api/model/vect.pkl', 'rb'))

@api_view(["POST"])
def resume_view(request):
    if request.method == 'POST':
        serializer = ResumeSerializer(data=request.data, context={"request": request})
        if serializer.is_valid():
            resume = [clean_resume(serializer.validated_data["resume"])]
            resume_vect = loaded_vect.transform(resume)
            answer = loaded_model.predict(resume_vect)[0]
            return Response(status=status.HTTP_200_OK, data={"answer":answer})
        return Response(status=status.HTTP_400_BAD_REQUEST)


