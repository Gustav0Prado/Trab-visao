# Visão Computacional e Percepção 👀
Repositório dedicado para os trabalhos da disciplina de Visão Computacional e Percepção na UFPR (ci1026).
## Trabalho 1 - Texturas
Escolha um tipo de imagem para segmentar por textura. Por exemplo, pode ser imagem médica, paisagem, urbana, equipamentos ou agrícola.
Monte um conjunto de filtros de textura, ao menos coma as orientações horizontal, vertical, 45 e 135 graus, e circular, processe a imagem em 3 escalas.
Você pode fazer os filtros em 3 escalas e aplicá-los nas imagens ou trabalhar com os filtros em uma escala e sucessivamente aplicar um filtro gaussiano e reduzir em seguida a imagem à metade em cada dimensão.
Construa vetores que descrevem cada região em termos de textura, sendo a média de cada janela (ver slides).
Tente fazer algo que agrupe os semelhantes, por exemplo baseado na distância euclidiana N-dimensional.
Mostre imagem com a categorização conforme grupso definidos.
Não precisa ser genérico, é o primeiro trabalho. Basta funcionar para um conjunto de ao menos 16 imagens que você vai analisar.
Use algoritmos clássicos, sem deep learning.
Pode ser feito com 2 a 4 participantes por grupo. Se estiver fora desta faixa ok, mas recebe 25% de desconto na nota por participante excedente ou faltante.
Pode ser em Python e OpenCV ou usar Colab. Pode ser em C ou C++ tb.

**Entrega: relatorio formato artigo mostrando o que vc fez, algo simples. Deve conter link clicável para o git da tarefa.**
