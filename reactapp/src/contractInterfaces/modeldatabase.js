import web3 from './web3';
import { modelDatabase} from "../abi/abi";

const address = '0x5f7Be1808d71Eb741B608a2b3a6Ca9996c4250D5';

export default new web3.eth.Contract(modelDatabase, address);