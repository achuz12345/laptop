"""
URL configuration for lap project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from django.conf import settings
from django.conf.urls.static import static
from lapapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('lapapp.urls')),
    path('',views.first),
    path('products/', views.product_list, name='product_list'),
    path('data/login/',views.home),
    path('login/',views.login_view),
    path('add-product/',views.add_product),
    path('edit/<int:id>/', views.edit_product, name='edit_product'),
    path('delete-product/<int:id>/',views.delete_product),
    path("register/", views.register_request, name="register"),
    path('cart/', views.view_cart, name='view_cart'),
    path('add/<int:product_id>/', views.add_to_cart, name='add_to_cart'),
    path('remove/<int:item_id>/', views.remove_from_cart, name='remove_from_cart'),
    path('checkout/', views.checkout, name='checkout'),
    path('add_category/', views.add_category, name='add_category'),
    path('confirm-purchase/', views.confirm_purchase, name='confirm_purchase'),
    path('thank-you/', views.thank_you_page, name='thank_you_page'),
    path('product/<int:product_id>/', views.product_detail, name='product_detail'),
    path('supplier_home/',views.supplier_home, name='supplier_home'),
    
    
    
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
