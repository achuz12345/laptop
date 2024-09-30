from django import forms
from .models import Product,CustomerProfile,Orders, Supplier
from django.contrib.auth.models import User
import re
from django.core.validators import RegexValidator
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _



class ProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = ['name', 'description', 'price', 'category', 'manufacture', 'quantity', 'os', 'ram', 'processor', 'image']



from django import forms
from django.contrib.auth.models import User
from .models import UserProfileInfo

class UserForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    confirm_password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ('username', 'email', 'password')

class UserProfileInfoForm(forms.ModelForm):
    class Meta:
        model = UserProfileInfo
        fields = ('email',)

# forms.py
from django import forms
from .models import Product, CartItem

class CheckoutForm(forms.ModelForm):
    phone_validator = RegexValidator(
        regex=r'^[6-9]{1}[0-9]{9}$',
        message='Phone number must start with a digit between 6 and 9, followed by 9 digits.'
    )

    phone = forms.CharField(
        label='contact',
        max_length=45,
        validators=[phone_validator]
    )
    def clean_phone(self):
        phone = self.cleaned_data['contact']
        return phone
    class Meta:
        model = Orders
        fields = ('fname','lname','address','contact','total')
    
class CoustomerForm(forms.ModelForm):
    password = forms.CharField(
        label=_('Password'),
        widget=forms.PasswordInput(),
        max_length=20,
        help_text=_("Your password must contain at least one uppercase letter, one lowercase letter, one digit, and one special character."),
    )
    def clean_password(self):
        password = self.cleaned_data.get('password')

        try:
            validate_password(password)
        except ValidationError as e:
            raise forms.ValidationError(e.messages)

        if not re.search(r'[A-Z]', password):
            raise forms.ValidationError(_('The password must contain at least one uppercase letter.'))

        if not re.search(r'[a-z]', password):
            raise forms.ValidationError(_('The password must contain at least one lowercase letter.'))

        if not re.search(r'[0-9]', password):
            raise forms.ValidationError(_('The password must contain at least one digit.'))

        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            raise forms.ValidationError(_('The password must contain at least one special character.'))

        return password



    email_validator = RegexValidator(
        regex=r'^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$',
        message='Enter a valid email address with the pattern: user@example.com'
    )
    
    email = forms.CharField(
        label='Email',
        max_length=20,
        validators=[email_validator]
    )

    def clean_email(self):
        email = self.cleaned_data['email']
        return email
    class Meta:
        model = CustomerProfile
        fields = "__all__"
    
    
    phone_validator = RegexValidator(
        regex=r'^[6-9]{1}[0-9]{9}$',
        message='Phone number must start with a digit between 6 and 9, followed by 9 digits.'
    )

    phone = forms.CharField(
        label='Phone',
        max_length=45,
        validators=[phone_validator]
    )
    def clean_phone(self):
        phone = self.cleaned_data['phone']
        return phone



from .models import SupplyRequest
from django.contrib.auth.hashers import make_password
class SupplyRequestForm(forms.ModelForm):
    class Meta:
        model = SupplyRequest
        fields = ['product', 'supplier', 'quantity']
class SupplierForm(forms.ModelForm):
    class Meta:
        model = Supplier
        fields = ['name', 'username', 'phone', 'email', 'address', 'password']

    def save(self, commit=True):
        supplier = super().save(commit=False)
        supplier.password = make_password(self.cleaned_data['password'])
        if commit:
            supplier.save()
        return supplier
    
from django import forms
from django.contrib.auth.hashers import check_password

class LoginForm(forms.Form):
    username = forms.CharField(max_length=255)
    password = forms.CharField(widget=forms.PasswordInput)

    def clean(self):
        cleaned_data = super().clean()
        username = cleaned_data.get('username')
        password = cleaned_data.get('password')

        supplier = Supplier.objects.filter(username=username).first()
        if supplier and check_password(password, supplier.password):
            return cleaned_data
        else:
            raise forms.ValidationError('Invalid username or password')