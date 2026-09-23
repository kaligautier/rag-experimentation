"""Parent contexts must remain linked to complete, searchable leaf sections."""

from app.adapters.parsing.chunker import HierarchicalChunker


async def should_build_parent_contexts_with_bidirectional_links():
    period = "### Période\n- Juin à septembre"
    equipment = "### Équipement\n- Chaussures\n- Sac à dos"
    source = f"# GR20\n\n## Conseils\n\n{period}\n\n{equipment}"

    nodes = await HierarchicalChunker().split(source)

    assert len(nodes) == 4
    root, advice, when, gear = nodes
    assert root.content == source
    assert advice.content == f"## Conseils\n\n{period}\n\n{equipment}"
    assert root.hierarchy.child_ids == [advice.id]
    assert advice.hierarchy.parent_id == root.id
    assert advice.hierarchy.child_ids == [when.id, gear.id]
    assert when.hierarchy.parent_id == gear.hierarchy.parent_id == advice.id
    assert [node.hierarchy.level for node in nodes] == [0, 1, 2, 2]
    assert [node.is_leaf for node in nodes] == [False, False, True, True]
    assert [when.content, gear.content] == [period, equipment]


async def should_keep_parent_introduction_searchable_as_a_leaf():
    introduction = "# GR20\n\nPréparation indispensable."
    section = "## Équipement\n\nPrendre des chaussures."

    nodes = await HierarchicalChunker().split(f"{introduction}\n\n{section}")

    assert [node.content for node in nodes if node.is_leaf] == [introduction, section]
    assert len(nodes[0].hierarchy.child_ids) == 2
    assert all(node.hierarchy.parent_id == nodes[0].id for node in nodes[1:])


async def should_split_long_sections_using_the_configured_token_budget():
    source = "# Guide\n\n" + " ".join(
        f"Etape {i} : préparer le sac et vérifier la météo." for i in range(30)
    )
    chunker = HierarchicalChunker(chunk_size=80, chunk_overlap=10)

    nodes = await chunker.split(source)

    leaves = [node for node in nodes if node.is_leaf]
    assert len(leaves) > 1
    assert nodes[0].content == source
    assert all(node.hierarchy.parent_id == nodes[0].id for node in leaves)
    assert all(node.hierarchy.token_count <= 80 for node in leaves)
    assert "Etape 0" in leaves[0].content
    assert "Etape 29" in leaves[-1].content
