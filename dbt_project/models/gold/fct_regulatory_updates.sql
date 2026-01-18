{{
    config(
        materialized='table',
        schema='analytics',
        unique_id='document_number'
    )
}}

with source_data as (
    select
        document_number,
        title,
        abstract,
        publication_date,
        html_url,
        is_ai,
        ai_reasoning,
        ai_confidence,
        ingestion_timestamp
    from {{ source('raw', 'raw_regulations') }}
)

, cleaned_abstract as (
    select
        document_number,
        title,
        html_url,
        -- Clean abstract: remove common HTML tags and extra whitespace
        trim(
            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        abstract,
                        '<[^>]*>',  -- Remove HTML tags
                        ''
                    ),
                    '\s+',  -- Replace multiple spaces with single space
                    ' '
                ),
                '&nbsp;|&lt;|&gt;|&amp;|&quot;|&#39;',  -- Remove HTML entities
                ''
            )
        ) as abstract_cleaned,
        publication_date,
        is_ai,
        ai_reasoning,
        ai_confidence,
        ingestion_timestamp
    from source_data
)

, risk_level_assignment as (
    select
        document_number,
        title,
        html_url,
        abstract_cleaned,
        publication_date,
        is_ai,
        ai_reasoning,
        ai_confidence,
        ingestion_timestamp,
        case
            when lower(abstract_cleaned) like '%penalty%' 
                or lower(abstract_cleaned) like '%enforcement%' 
                or lower(abstract_cleaned) like '%prohibition%'
                then 'High'
            when lower(abstract_cleaned) like '%compliance%' 
                or lower(abstract_cleaned) like '%requirement%'
                then 'Medium'
            else 'Low'
        end as risk_level
    from cleaned_abstract
)

select
    document_number,
    title,
    html_url,
    abstract_cleaned as abstract,
    risk_level,
    publication_date,
    is_ai,
    ai_reasoning,
    ai_confidence,
    current_timestamp as dbt_updated_at
from risk_level_assignment
