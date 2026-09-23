"""Markdown sections retain their body and inherit ancestor heading context."""

import pytest

from app.adapters.parsing.chunker import HierarchicalChunker


async def should_split_markdown_gr20_into_two_complete_sections():
    period = """### Meilleure Période

- De juin à septembre pour des conditions favorables.
- Juillet et août sont les mois les plus fréquentés.
- Juin et septembre offrent davantage de tranquillité."""
    equipment = """### Équipement Essentiel

- Chaussures de randonnée robustes.
- Sac à dos adapté à une itinérance de plusieurs jours.
- Vêtements imperméables et chauds.
- Carte, boussole et GPS.
- Réserve d'eau et nourriture."""
    markdown = f"# GR20\n\n## Conseils Pratiques\n\n{period}\n\n{equipment}"

    chunks = await HierarchicalChunker().split(markdown)

    chunks = [chunk for chunk in chunks if chunk.is_leaf]
    assert len(chunks) == 2
    chunks = [chunk for chunk in chunks if chunk.is_leaf]
    assert [chunk.content for chunk in chunks] == [period, equipment]
    assert [chunk.metadata for chunk in chunks] == [
        {
            "header_path": "/GR20/Conseils Pratiques/",
            "section_title": "Meilleure Période",
            "header_level": 3,
        },
        {
            "header_path": "/GR20/Conseils Pratiques/",
            "section_title": "Équipement Essentiel",
            "header_level": 3,
        },
    ]
    for chunk in chunks:
        assert chunk.embedding_text.startswith("/GR20/Conseils Pratiques/")
        assert chunk.embedding_text.endswith(chunk.content)
        assert chunk.embedding_text.count(chunk.content) == 1


async def should_split_markdown_without_leaking_ancestor_context_between_branches():
    equipment = "### Équipement\n\nEmporter des chaussures de randonnée."
    route = "### Étapes\n\nCommencer à Calenzana."
    alternative = "### Étapes\n\nTraverser les Pyrénées."
    markdown = (
        f"# GR20\n\n## Conseils\n\n{equipment}\n\n"
        f"## Itinéraire\n\n{route}\n\n# GR10\n\n{alternative}"
    )

    chunks = await HierarchicalChunker().split(markdown)

    chunks = [chunk for chunk in chunks if chunk.is_leaf]
    assert [chunk.content for chunk in chunks] == [equipment, route, alternative]
    assert [chunk.metadata["header_path"] for chunk in chunks] == [
        "/GR20/Conseils/",
        "/GR20/Itinéraire/",
        "/GR10/",
    ]
    assert chunks[-1].embedding_text.startswith("/GR10/")
    assert "GR20" not in chunks[-1].embedding_text
    assert "Conseils" not in chunks[1].embedding_text


async def should_split_markdown_with_a_heading_level_jump():
    section = "### Ravitaillement\n\nPrévoir de l'eau pour chaque étape."

    chunks = await HierarchicalChunker().split(f"# GR20\n\n{section}")

    chunks = [chunk for chunk in chunks if chunk.is_leaf]
    assert len(chunks) == 1
    assert chunks[0].content == section
    assert chunks[0].metadata == {
        "header_path": "/GR20/",
        "section_title": "Ravitaillement",
        "header_level": 3,
    }
    assert chunks[0].embedding_text.startswith("/GR20/")


async def should_split_markdown_preserving_parent_intro_before_child_sections():
    introduction = "# GR20\n\nLe GR20 traverse la Corse du nord au sud."
    advice = "## Conseils\n\nUne préparation physique est recommandée."

    chunks = await HierarchicalChunker().split(f"{introduction}\n\n{advice}")

    chunks = [chunk for chunk in chunks if chunk.is_leaf]
    assert [chunk.content for chunk in chunks] == [introduction, advice]
    assert chunks[0].metadata == {
        "header_path": "/",
        "section_title": "GR20",
        "header_level": 1,
    }
    assert chunks[1].metadata == {
        "header_path": "/GR20/",
        "section_title": "Conseils",
        "header_level": 2,
    }


async def should_split_markdown_preserving_tables_and_headings_inside_fenced_code():
    commands = """## Commandes

Voici les commandes utiles :

| Commande | Effet |
| --- | --- |
| `pwd` | Affiche le répertoire |

```sh
# Ceci est un commentaire, pas un titre
printf '%s\\n' '# Toujours du code'
```

Le paragraphe après le code appartient à cette section."""
    summary = "## Résumé\n\nConserver toute la structure Markdown."
    markdown = f"# Guide\n\n{commands}\n\n{summary}"

    chunks = await HierarchicalChunker().split(markdown)

    chunks = [chunk for chunk in chunks if chunk.is_leaf]
    assert [chunk.content for chunk in chunks] == [commands, summary]
    assert [chunk.metadata["section_title"] for chunk in chunks] == [
        "Commandes",
        "Résumé",
    ]
    assert [chunk.metadata["header_path"] for chunk in chunks] == [
        "/Guide/",
        "/Guide/",
    ]
    assert chunks[0].embedding_text.endswith(commands)


@pytest.mark.parametrize(
    "markdown",
    [
        "",
        " \n\t\n",
        "# GR20",
        "# GR20\n\n## Conseils Pratiques\n\n### Meilleure Période\n",
    ],
)
async def should_split_markdown_without_emitting_empty_or_heading_only_chunks(markdown):
    chunks = await HierarchicalChunker().split(markdown)

    assert chunks == []


async def should_report_heading_token_limit_with_the_budget():
    from app.utils.error import DocumentLimitError

    chunker = HierarchicalChunker(chunk_size=16, chunk_overlap=0)
    with pytest.raises(DocumentLimitError) as caught:
        await chunker.split("# " + "heading " * 40 + "\n\nBody text.")
    assert caught.value.status_code == 413
    assert caught.value.details["limit_tokens"] == 16
    assert caught.value.details["heading_tokens"] >= 16
