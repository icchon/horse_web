from django.db import models
from django.utils import timezone

class Employee(models.Model):
  name = models.CharField("名前", max_length=255)
  mail = models.EmailField("メール", max_length=255)
  gender = models.BooleanField("性別")
  department = models.CharField("部署", max_length=255)
  year = models.IntegerField("社歴", default=0)
  created_at = models.DateField("作成日", default=timezone.now)
  
  def __str__(self):
    return "<id:" + str(self.id) + "," + self.name + "," + self.department + ">"