let doc_json = getData();


let info = new Vue({
  el: '#train-annotations',
  delimiters: ['[[', ']]'],
  data: {
    selected_concept: doc_json['annotations'] == undefined ? { "msg": "No documents yet" } : doc_json['annotations'][0],
    show: false,
    elementVisible: false,
    msg: ''
  },
  methods: {
    show_info: function(id) {
      this.selected_concept = doc_json['annotations'][id]
    },
    concept_feedback: function(neg) {
      if(neg){
        this.showmsg("Negative feedback recorded");
      }
      else{
        this.showmsg("Positive feedback recorded");
      }
      let d = {};
      d['cui'] = this.selected_concept['cui'];
      d['text'] = doc_json['text'];
      d['negative'] = neg;
      d['tkn_inds'] = [this.selected_concept['start_tkn'], this.selected_concept['end_tkn']];
      d['char_inds'] = [this.selected_concept['start_ind'], this.selected_concept['end_ind']];
      d['ajaxRequest'] = true;

      this.$http.post('/add_cntx', d, {
         headers: {
                 'X-CSRFToken': Cookies.get('csrftoken')
               }
      });
    },
    create_concept: function() {
      let d = {};
      console.log(this.$refs)
      d['name'] = this.$refs.cntx_name.value;
      d['cui'] = this.$refs.cui.value;
      d['tui'] = this.$refs.tui.value;
      d['source_value'] = this.$refs.source_value.value;
      d['synonyms'] = this.$refs.synonyms.value;
      d['text'] = this.$refs.cntx_text.value;
      d['ajaxRequest'] = true;

      this.$http.post('/add_concept_manual', d, {
         headers: {
                 'X-CSRFToken': Cookies.get('csrftoken')
               }
      });
      this.show=false;
      this.showmsg("New concept created");
    },

    save_cdb_model: function() {
      let d = {};
      d['ajaxRequest'] = true;
      this.$http.post('/save_cdb_model', d, {
         headers: {
                 'X-CSRFToken': Cookies.get('csrftoken')
               }
      });
      this.showmsg("New training data added and model saved");
    },

    reset_cdb_model: function() {
      let d = {};
      d['ajaxRequest'] = true;

      this.$http.post('/reset_cdb_model', d, {
         headers: {
                 'X-CSRFToken': Cookies.get('csrftoken')
               }
      });
      this.showmsg("Model reset to the last saved instance");
    },


    showmsg: function(msg) {
      this.msg = msg;
      this.elementVisible = true;
      let vm = this;
      setTimeout(function () { vm.hidemsg() }, 4000);
    },

    hidemsg: function() {
      this.elementVisible = false;
      console.log("HERE");
    }
 
  }
})
