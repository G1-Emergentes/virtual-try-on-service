-- Run in the Supabase project dedicated to Virtual Try-On models.
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values (
    'accessory-models', 'accessory-models', false, 52428800,
    array['model/gltf-binary', 'image/png', 'image/jpeg', 'image/webp', 'application/json']
)
on conflict (id) do nothing;
