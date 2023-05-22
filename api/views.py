import os
from glob import glob

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from api.serializers import ResumeSerializer
from api.services import clean_resume, get_suggestion
import joblib
import random
import PyPDF2

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
            return Response(status=status.HTTP_200_OK, data={"answer": answer, "suggestion": get_suggestion()})
        return Response(status=status.HTTP_400_BAD_REQUEST)


@api_view(["POST"])
def pdf_decode_view(request):
    for file in glob('tmp*.txt'):
        os.remove(file)
    if request.method == 'POST':
        file = request.FILES['pdf']
        file_name = "tmp" + str(random.randint(1000, 9999)) + ".txt"
        with open(file_name, "wb+") as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        file_obj = open(file_name, 'rb')
        pdf = PyPDF2.PdfReader(file_obj)
        text = ""
        for i in range(0, len(pdf.pages)):
            text = text + '\n\n' + pdf.pages[i].extract_text()
        file_obj.close()
        resume = [clean_resume(text)]
        resume_vect = loaded_vect.transform(resume)
        answer = loaded_model.predict(resume_vect)[0]
        return Response(status=status.HTTP_200_OK, data={"answer": answer, "suggestion": get_suggestion()})
    return Response(status=status.HTTP_400_BAD_REQUEST)
