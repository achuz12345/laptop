from django.conf.urls.static import static
from django.conf import settings
from django.urls import path
from django.contrib import admin
from .views import * 
urlpatterns = [
    path('', first, name='first'),
    path('products/', product_list, name='product_list'),
    path('admin/', admin.site.urls),
    path('about/', about, name='about'),
    path('login/', login_view, name='login'), 
    path('logout/', logout_view, name='logout'),
    path('alogout/', alogout_view, name='adminlogout'),
    path('home/', home, name='home'),
    path('add-product/', add_product, name='add_product'),
    path('edit-product/<int:id>/', edit_product, name='edit_product'),
    path('delete-product/<int:id>/', delete_product, name='delete_product'),
    path("register", register_request, name="register"),
    path("forgot", forgot_password, name="forgot"),
    path("forgot1",forgot1),
    path('password_new',password_new),
    path('myprofile',myprofile,name='view_profile'),
    path('edit_profile',edit_profile,name='edit_profile'),
    path('cart/', view_cart, name='view_cart'),
    path('orders',orders,name='orders'),
    path('all_orders',all_orders,name='all_orders'),
    path('ordered_items/<str:id>',ordered_items,name='ordered_items'),
    path('aordered_items/<str:id>',aordered_items,name='aordered_items'),
    path('ship_order/<str:id>',ship_order,name='ship_order'),
    path('deliver_order/<str:id>',deliver_order,name='deliver_order'),

    path('add/<int:product_id>/', add_to_cart, name='add_to_cart'),
    path('remove/<int:item_id>/',remove_from_cart, name='remove_from_cart'),
    path('index/', index_home, name='index'),
    path('checkout/', checkout, name='checkout'),
    path('confirm-purchase/', confirm_purchase, name='confirm_purchase'),
    path('add_category/', add_category, name='add_category'),
    path('confirm-purchase/', confirm_purchase, name='confirm_purchase'),
    path('thank-you/', thank_you_page, name='thank_you_page'),
    path('product/<int:product_id>/', product_detail, name='product_detail'),
    path('add_supplier/', add_supplier, name='add_supplier'),
    path('send_supply_request/', send_supply_request, name='send_supply_request'),
    path('view_supply_requests/', view_supply_requests, name='view_supply_requests'),
    path('manage_supply_request/<int:request_id>/', manage_supply_request, name='manage_supply_request'),
    path('supplier_home/',supplier_home, name='supplier_home'),
    path('supplier/login/', supplier_login_view, name='supplier_login'),
    path('chatbot/', chatbot, name='chatbot'),
    path('visualization', dashboard, name='dashboard'),
    path('admin-reports/', admin_reports, name='admin_reports'),
    path('download-report/', download_csv_report, name='download_csv_report'),
    path('recommendation/', laptop_recommendation, name='recommendation'),

]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
